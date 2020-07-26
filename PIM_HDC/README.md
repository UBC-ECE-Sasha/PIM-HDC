# PIM HDC

Hyperdimensional computing accelerator using processing in-memory.

## Usage

```text
usage: ./pim_hdc [ -d ] -i <INPUT_FILE>
        d: use DPU
        i: input file
        r: show runtime only
        s: show results
        t: test results      - Compare host against DPU
        h: help message
```

Example: Run HDC on DPU and host and verify they are the same:

```shell script
./pim_hdc -i data/large-data.bin -t
```

## Compile options

The following are compile time settable options and their defaults:

* `NR_DPUS = 32`
* `NR_TASKLETS = 1`

* `SHOW_DPU_LOGS = 1` - Print DPU logs to `stdout`
* `BULK_XFER = 1`     - Use bulk transfers to read and write data to DPU

The following should not be reduced unless a non-default dataset is used,
and they already represent the exact data sizes:

* `MAX_BIT_DIM = 313`
* `MAX_CHANNELS = 4`
* `MAX_N = 5`
* `MAX_IM_LENGTH = 22`
* `MAX_INPUT = 384`

## Implementation

The following is a high-level description of the hyperdimensional accelerator.

1. IM and CIM

   * A item memory (IM) maps all symbols in the system to the HD space.

     In the case where `iM` is kept in MRAM (the default), it is defined in in [src/dpu/dpu_task.c](src/dpu/dpu_task.c).

     When kept in WRAM, IM is defined in [include/init.h](include/init.h):

     IM can be thought of as a multi-dimensional array:

     ```C
     uint32_t iM[MAX_IM_LENGTH][MAX_BIT_DIM + 1];
     ```

     For ease-of-use with the DPU, it is kept in the form of a single dimensional array:

     ```C
     uint32_t iM[MAX_IM_LENGTH * MAX_BIT_DIM + 1)];
     ```

     It is pre-computed.

     The IM assigns a random hypervectors (with i.i.d. components) to every channel's name.

     IM represents discrete signals.

   * A continuous item memory (CIM) extends the notion of item memory to analog values (e.g., the signal levels of channels) for mapping.

     In the case where CIM is kept in MRAM (the default), it is defined in in [src/dpu/dpu_task.c](src/dpu/dpu_task.c).

     When kept in WRAM, CIM is defined in [include/init.h](include/init.h):

     CIM can be thought of as a multi-dimensional array:

     ```C
     uint32_t chAM[MAX_CHANNELS][MAX_BIT_DIM + 1]
     ```

     For ease-of-use with the DPU, it is kept in the form of a single dimensional array:

     ```C
     uint32_t chAM[MAX_CHANNELS * MAX_BIT_DIM + 1)]
     ```

     It is pre-computed.

     In the continuous vector space of CIM, orthogonal endpoint hypervectors are generated for the minimum and maximum signal levels.

   * The IM and CIM stay fixed throughout the computation, and they serve as seeds from which further representations are made.

2. The seed hypervectors are encoded by the MAP operations to represent the event of interest for classification.

   * A spatial encoder can represent a set of all channel-value pairs at timestamp t into a binary hypervector.

     The multiplication is used to bind each channel to its signal level, and to form the set all these bound hypervectors are bundled by the addition by the `compute_N_gram` function.

   * A temporal encoder can capture the relevant temporal information by using the permutation and multiplication that together form an N-gram hypervector from a sequence of N hypervectors. Hence, a sequence of N spatial hypervectors at consecutive timestamps are encoded into an N-gram hypervector.

3. For a given class, across all its trials, the corresponding N-gram hypervectors are added to produce a binary prototype hypervector.

   During training, the prototype hypervectors are stored in an associative memory (AM) as the learned patterns.

   During classification, in an identical way to prototypes, a query hypervector is generated from unseen inputs. The AM compares the query hypervectors to all learned prototype hypervectors, and returns the label of the one that has the minimum Hamming distance via the function `associative_memory_32bit`.

4. The classification result is output.
