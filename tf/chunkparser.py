#!/usr/bin/env python3
#
#    This file is part of Leela Chess.
#    Copyright (C) 2018 Folkert Huizinga, et al.

import itertools
import multiprocessing as mp
import numpy as np
import random
import shufflebuffer as sb
import struct
import gzip
from time import sleep
from select import select

V6_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffffffffffffIHHff"
v6_struct = struct.Struct(V6_STRUCT_STRING)

def chunk_reader(chunk_filenames, chunk_filename_queue):
    chunks, done = [], list(chunk_filenames)
    while True:
        if not chunks: chunks, done = done, []; random.shuffle(chunks)
        if not chunks: done = list(chunk_filenames); continue
        chunk_filename_queue.put(chunks.pop())

def reverse_expand_bits(plane):
    """ Helper function to safely expand a byte into a plane. """
    return np.unpackbits(np.array([plane], dtype=np.uint8))[::-1].astype(np.float32).tobytes()

def convert_v6_to_tuple(content):
    """Unpacks a v6 binary record into a 9-tuple for TensorFlow."""
    
    (ver, input_format, probs, planes, us_ooo, us_oo, them_ooo, them_oo,
     stm, rule50_count, invariance_info, dep_result, root_q, best_q, root_d,
     best_d, root_m, best_m, plies_left, result_q, result_d, played_q,
     played_d, played_m, orig_q, orig_d, orig_m, visits, played_idx,
     best_idx, pol_kld, entropy) = v6_struct.unpack(content[:v6_struct.size])
    
    if plies_left == 0: plies_left = invariance_info
    plies_left_packed = struct.pack("f", plies_left)
    
    planes_unpacked = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.float32)
    rule50_divisor = 99.0 if input_format < 4 else 100.0
    rule50_plane = struct.pack("f", rule50_count / rule50_divisor) * 64
    
    # --- THIS IS THE DEFINITIVE FIX FOR THE IndexError ---
    # This robust logic is now used for ALL input formats, removing the faulty if/else block.
    # It safely expands byte values into planes instead of using them as indices.
    them_ooo_bytes, us_ooo_bytes = reverse_expand_bits(them_ooo), reverse_expand_bits(us_ooo)
    them_oo_bytes, us_oo_bytes = reverse_expand_bits(them_oo), reverse_expand_bits(us_oo)
    enpassant_bytes = reverse_expand_bits(stm)
    # 32 bytes per plane (float32)
    middle_planes = (us_ooo_bytes + us_oo_bytes + them_ooo_bytes + them_oo_bytes + enpassant_bytes).ljust(7 * 64 * 4, b'\0')

    all_planes = (planes_unpacked.tobytes()[:104*64*4] + middle_planes + rule50_plane).ljust(112 * 64 * 4, b'\0')
    
    winner = struct.pack("fff", float(dep_result == 1.0), float(dep_result == 0.0), float(dep_result == -1.0))
    def qd_to_wdl(q,d): q,d=min(max(q,-1.),1.),min(max(d,0.),1.); return (0.5*(1-d+q),d,0.5*(1-d-q))
    root_wdl, st_wdl = struct.pack("fff", *qd_to_wdl(root_q, root_d)), struct.pack("fff", *qd_to_wdl(root_q, root_d))
    
    return (all_planes, probs, winner, root_wdl, plies_left_packed, st_wdl, b'', b'', b'')

class ChunkParser:
    def __init__(self, chunks, expected_input_format, shuffle_size=1, sample=1, **kwargs):
        self.inner = ChunkParserInner(self, chunks, shuffle_size, sample, **kwargs)
    def shutdown(self):
        if hasattr(self, 'processes'):
            for p in self.processes: p.terminate(); p.join()
        if hasattr(self.inner, 'readers'):
            for r, w in zip(self.inner.readers, self.inner.writers): r.close(); w.close()
        if hasattr(self, 'chunk_process'):
            self.chunk_process.terminate(); self.chunk_process.join()
    def parse(self): return self.inner.parse()

class ChunkParserInner:
    def __init__(self, parent, chunks, shuffle_size, sample, batch_size=256, workers=None, **kwargs):
        self.sample, self.batch_size, self.shuffle_size, self.record_size = sample, batch_size, shuffle_size, 24756
        print(f"INFO: Using hard-coded data record size of {self.record_size} bytes.")
        if workers is None: workers = max(1, mp.cpu_count() - 2)
        if workers > 0:
            print(f"Using {workers} worker processes.")
            self.readers, self.writers, parent.processes = [], [], []
            self.chunk_filename_queue = mp.Queue(maxsize=4096)
            for _ in range(workers):
                read, write = mp.Pipe(duplex=False); p = mp.Process(target=self.task, args=(self.chunk_filename_queue, write))
                p.daemon = True; parent.processes.append(p); p.start(); self.readers.append(read); self.writers.append(write)
            parent.chunk_process = mp.Process(target=chunk_reader, args=(chunks, self.chunk_filename_queue))
            parent.chunk_process.daemon = True; parent.chunk_process.start()
        else: self.chunks = chunks
    def sample_and_slice_records(self, chunkdata):
        for i in range(0, len(chunkdata), self.record_size):
            if len(chunkdata)-i < self.record_size: continue
            if self.sample > 1 and random.randint(0, self.sample-1) != 0: continue
            yield chunkdata[i : i+self.record_size]
    def single_file_gen(self, filename):
        try:
            with gzip.open(filename, "rb") as f:
                chunkdata = f.read()
                if not chunkdata: return
                for item in self.sample_and_slice_records(chunkdata): yield item
        except Exception as e: print(f"Warning: Could not process {filename}. Error: {e}")
    def data_gen(self):
        sbuff = sb.ShuffleBuffer(self.record_size, self.shuffle_size)
        while self.readers:
            for r in select(self.readers,[],[],0.01)[0]:
                try:
                    s = r.recv_bytes()
                    if len(s) != self.record_size: continue
                    s_out = sbuff.insert_or_replace(s)
                    if s_out: yield s_out
                except (EOFError, ConnectionResetError):
                    self.readers.remove(r)
        while True: 
            s = sbuff.extract()
            if s is None:
                return
            yield s
    def tuple_gen(self, gen):
        for r in gen: yield convert_v6_to_tuple(r)
    def batch_gen(self, gen):
        while True:
            s=list(itertools.islice(gen, self.batch_size))
            if not s: return
            yield tuple(map(b"".join, zip(*s)))
    def parse(self):
        gen = self.data_gen(); gen = self.tuple_gen(gen); gen = self.batch_gen(gen)
        for b in gen: yield b
    def task(self, queue, writer):
        while True:
            try:
                filename = queue.get(timeout=10)
                for item in self.single_file_gen(filename): writer.send_bytes(item)
            except (KeyboardInterrupt, EOFError):
                break
