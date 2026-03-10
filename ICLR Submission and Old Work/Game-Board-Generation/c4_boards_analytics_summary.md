# C4 Game Board Analytics

Definitions used for reviewer feedback:
- Circuit: any root-to-node path in the game-board DAG.
- Efficient/optimal circuit: a shortest root-to-node path.
- Multiple optimal circuits: nodes with `shortest_path_count > 1`.

## C4-main-multivar

- Nodes: 20,000
- Edges: 31,746
- Roots: 3
- Nodes with multiple optimal circuits: 18,966 (94.83%)
- Max number of optimal circuits for a single node: 164
- Max number of total circuits for a single node: 1,596
- Step vs shortest-depth mismatches: 0

### Depth Distribution (step)
- depth 0: 3
- depth 1: 12
- depth 2: 174
- depth 3: 10,862
- depth 4: 8,949

### Optimal Circuit Count Buckets
- 1: 1,034
- 2: 1,957
- 3-5: 8,894
- 6-10: 4,830
- >10: 3,285

### Top Nodes By Number of Optimal Circuits
- `8*x0**4*x1*x2` (step=4, optimal=164, total=669)
  example shortest path: x0 -> x0**2 -> x0**3 -> 2*x0**3*x1 -> 8*x0**4*x1*x2
- `6*x0**3*x1*x2` (step=4, optimal=138, total=1,534)
  example shortest path: x0 -> 2*x0 -> 3*x0 -> 3*x0**2 -> 6*x0**3*x1*x2
- `2*x0**2 + 3*x0 + x1 + x2` (step=4, optimal=120, total=407)
  example shortest path: x0 -> x0**2 -> x0**2 + x2 -> 2*x0**2 + x2 -> 2*x0**2 + 3*x0 + x1 + x2
- `6*x0 + 2*x1 + x2` (step=4, optimal=112, total=496)
  example shortest path: x0 -> 2*x0 -> 3*x0 -> 4*x0 + x1 -> 6*x0 + 2*x1 + x2
- `6*x0 + x1 + 2*x2` (step=4, optimal=112, total=496)
  example shortest path: x0 -> 2*x0 -> 3*x0 -> 4*x0 + x1 -> 6*x0 + x1 + 2*x2
- `4*x0**5*x1*x2` (step=4, optimal=111, total=356)
  example shortest path: x0 -> x0**2 -> x0**3 -> 2*x0**4 -> 4*x0**5*x1*x2
- `5*x0 + 2*x1 + 2*x2` (step=4, optimal=108, total=459)
  example shortest path: x0 -> 2*x0 -> 3*x0 -> 3*x0 + 2*x1 -> 5*x0 + 2*x1 + 2*x2
- `x0**2 + 5*x0 + x1 + x2` (step=4, optimal=105, total=367)
  example shortest path: x0 -> 2*x0 -> 3*x0 -> x0**2 + 3*x0 -> x0**2 + 5*x0 + x1 + x2
- `7*x0 + x1 + x2` (step=4, optimal=104, total=452)
  example shortest path: x0 -> 2*x0 -> 3*x0 -> 5*x0 -> 7*x0 + x1 + x2
- `4*x0**4*x1**2*x2` (step=4, optimal=103, total=166)
  example shortest path: x0 -> x0**2 -> x0**3 -> 2*x0**3*x1 -> 4*x0**4*x1**2*x2

## C4-pretraining-singlevar

- Nodes: 6,656
- Edges: 20,966
- Roots: 1
- Nodes with multiple optimal circuits: 6,592 (99.04%)
- Max number of optimal circuits for a single node: 92
- Max number of total circuits for a single node: 15,296
- Step vs shortest-depth mismatches: 0

### Depth Distribution (step)
- depth 0: 1
- depth 1: 2
- depth 2: 9
- depth 3: 96
- depth 4: 6,548

### Optimal Circuit Count Buckets
- 1: 64
- 2: 225
- 3-5: 2,592
- 6-10: 2,453
- >10: 1,322

### Top Nodes By Number of Optimal Circuits
- `8*x**8 + 16*x**7` (step=4, optimal=92, total=11,290)
  example shortest path: x -> x**2 -> x**3 -> x**3 + 2*x**2 -> 8*x**8 + 16*x**7
- `4*x**9 + 8*x**8` (step=4, optimal=90, total=9,001)
  example shortest path: x -> x**2 -> x**3 -> x**5 -> 4*x**9 + 8*x**8
- `8*x**8 + 8*x**7` (step=4, optimal=89, total=7,036)
  example shortest path: x -> x**2 -> x**2 + x -> x**3 + x**2 -> 8*x**8 + 8*x**7
- `4*x**8 + 8*x**7` (step=4, optimal=85, total=8,882)
  example shortest path: x -> x**2 -> x**3 -> x**5 -> 4*x**8 + 8*x**7
- `8*x**7 + 16*x**6` (step=4, optimal=83, total=10,850)
  example shortest path: x -> x**2 -> x**3 -> x**3 + 2*x**2 -> 8*x**7 + 16*x**6
- `4*x**9 + 4*x**8` (step=4, optimal=83, total=4,707)
  example shortest path: x -> x**2 -> x**2 + x -> x**3 + x**2 -> 4*x**9 + 4*x**8
- `8*x**7 + 8*x**6` (step=4, optimal=80, total=6,567)
  example shortest path: x -> x**2 -> x**2 + x -> x**3 + x**2 -> 8*x**7 + 8*x**6
- `24*x**7` (step=4, optimal=79, total=4,681)
  example shortest path: x -> 2*x -> 3*x -> 3*x**2 -> 24*x**7
- `4*x**8 + 4*x**7` (step=4, optimal=78, total=4,582)
  example shortest path: x -> x**2 -> x**2 + x -> x**3 + x**2 -> 4*x**8 + 4*x**7
- `12*x**8` (step=4, optimal=77, total=2,392)
  example shortest path: x -> 2*x -> 3*x -> 3*x**2 -> 12*x**8
