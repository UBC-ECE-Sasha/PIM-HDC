# PIM HDC

Hyperdimensional computing accelerator using processing in-memory.

## Implementation

The following is a high-level description of the hyperdimensional accelerator.

1. IM and CIM

   * A item memory (IM) maps all symbols in the system to the HD space.

     IM is defined in `data.h` as `uint32_t iM[][BIT_DIM + 1]`. It is pre-computed.

     The IM assigns a random hypervectors (with i.i.d. components) to every channel's name.

     IM represents discrete signals.

   * A continuos item memory (CIM) extends the notion of item memory to analog values (e.g., the signal levels of channels) for mapping.

     CIM is defined in `data.h` as `uint32_t chM[CHANNELS][BIT_DIM + 1]`. It is pre-computed.

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
