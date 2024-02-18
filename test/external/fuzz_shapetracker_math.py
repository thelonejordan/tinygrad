import random
from typing import Tuple
from tqdm import trange
from tinygrad.helpers import getenv, DEBUG, colored
from tinygrad.shape.shapetracker import ShapeTracker, _project_view
from test.external.fuzz_shapetracker import shapetracker_ops
from test.external.fuzz_shapetracker import do_permute, do_reshape_split_one, do_reshape_combine_two, do_flip, do_pad
from test.unit.test_shapetracker_math import st_equal, MultiShapeTracker

def fuzz_plus(verbose=True) -> Tuple[ShapeTracker, ShapeTracker]:
  start = ShapeTracker.from_shape((random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)))
  m = MultiShapeTracker([start])
  if DEBUG >=2 and verbose and getenv("VERBOSE") == 2: print(f"{start=}")
  for _ in range(4): random.choice(shapetracker_ops)(m, verbose)
  backup = m.sts[0]
  if DEBUG >=2 and verbose and getenv("VERBOSE") == 2: print(f"{backup=}")
  m.sts.append(ShapeTracker.from_shape(m.sts[0].shape))
  for _ in range(4): random.choice(shapetracker_ops)(m, verbose)
  st_sum = backup + m.sts[1]
  return m.sts[0], st_sum

# shrink and expand aren't invertible, and stride is only invertible in the flip case
invertible_shapetracker_ops = [do_permute, do_reshape_split_one, do_reshape_combine_two, do_flip, do_pad]

def fuzz_invert(verbose=True) -> Tuple[ShapeTracker, ShapeTracker]:
  start = ShapeTracker.from_shape((random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)))
  m = MultiShapeTracker([start])
  if DEBUG >=2 and getenv("VERBOSE") == 2 and verbose: print(f"{start=}")
  for _ in range(8): random.choice(invertible_shapetracker_ops)(m, verbose)
  inv = m.sts[0].invert(start.shape)
  st_sum = (m.sts[0] + inv) if inv else None
  return start, st_sum

def _display_debug_info(exp: ShapeTracker, got: ShapeTracker):
  if len(exp.views) > 1:
    print()
    exp = ShapeTracker(exp.views[-2:])
    expv2, expv1 = exp.views
    orig, _, pstrides = _project_view(expv2, expv1)
    print(f"expv1.origin={tuple(orig)}")
    print(f"expv1.pstrides={tuple(pstrides)}")
    print(f"expv1.real_pstrides={exp.real_strides()}")
  if len(got.views) > 1:
    print()
    got = ShapeTracker(got.views[-2:])
    gotv2, gotv1 = got.views
    orig, _, pstrides = _project_view(gotv2, gotv1)
    print(f"gotv1.origin={tuple(orig)}")
    print(f"gotv1.pstrides={tuple(pstrides)}")
    print(f"gotv1.real_pstrides={got.real_strides()}")

if __name__ == "__main__":
  if seed:=getenv("SEED"): random.seed(seed)
  verbose = getenv("VERBOSE", 1) >= 1 and not getenv("ONLY_NEQ") if DEBUG >=1 else False
  total = getenv("CNT", 1000)
  for fuzz in [globals()[f'fuzz_{x}'] for x in getenv("FUZZ", "invert,plus").split(",")]:
    same_but_neq = same_but_neq_canon = same_but_neq_canon_repr = nviews = nviews_simpl = nviews_canon = 0
    for _ in trange(total, desc=f"{fuzz}"):
      # fuzz shapetrackers
      st1, st2 = fuzz(verbose)
      sts1, sts2 = st1.simplify(), st2.simplify()
      stc1, stc2 = st1.canonicalize(), st2.canonicalize()
      stcr1, stcr2 = ShapeTracker(stc1.views), ShapeTracker(stc2.views)
      # compare results
      eq = st_equal(st1, st2)
      eqs = sts1 == sts2
      eqc = stc1 == stc2
      eqcr = stcr1 == stcr2
      # update stats
      if eq and not eqs: same_but_neq += 1
      if eq and not eqc: same_but_neq_canon += 1
      if eq and not eqcr: same_but_neq_canon_repr += 1
      nviews += len(st1.views)
      nviews_simpl += len(sts1.views)
      nviews_canon += len(stc1.views)
      # print stuff
      filterout = (eqc if getenv("ONLY_NEQC") else eqs) if getenv("ONLY_NEQ") and DEBUG >=1 else False
      if not filterout:
        if eq and not eqc:
          if DEBUG >=2 and getenv("VERBOSE") == 2:
            # extra stuff for printing
            _display_debug_info(st1, st2)
        if DEBUG >=1 and getenv("CHECK_NEQ") and eq and not (eqs and eqc):
          print(f"EXP SIMPL: {sts1}")
          print(f"GOT SIMPL: {sts2}")
        if DEBUG >=2 and getenv("CHECK_NEQ") and eq and not (eqs and eqc):
          print(f"EXP CANON: {stc1}")
          print(f"GOT CANON: {stc2}")
        if DEBUG >=1:
          print(f"EXP: {st1}")
          print(f"GOT: {st2}")
        if getenv("CHECK_NEQ") and eq and not eqs:
          print(colored("same but unequal", "yellow"))
        if DEBUG >=1:
          print(colored(f"****{' (symbolic)' if DEBUG >=2 else ''}", "green" if eq else "red"))
        if DEBUG >=2:
          print(colored("**** (canon repr)", "green" if eqcr else "yellow" if eqc else "red"))
          print(colored("**** (canon)", "green" if eqc else "red"))
      # mandatory checks
      if eq and eqs: assert eqc and eqcr
      if not eq: exit(0)
    # print agg stats
    if getenv("CHECK_NEQ"):
      print(f"same but unequal {(same_but_neq/total)*100:.2f}%")
      if DEBUG >=2:
        print(f"same but unequal canon repr {(same_but_neq_canon_repr/total)*100:.2f}%")
        print(f"same but unequal canon {(same_but_neq_canon/total)*100:.2f}%")
        print(f"views per shapetracker {(nviews/total):.2f}(real), {(nviews_simpl/total):.2f}(simpl), {(nviews_canon/total):.2f}(canon)")
