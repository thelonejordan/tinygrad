# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
import functools, math
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Set, cast, Iterable
from tinygrad.helpers import merge_dicts, getenv, prod, DEBUG
from tinygrad.shape.symbolic import Variable, MulNode, Node, SumNode, NumNode, sint
from tinygrad.shape.view import View, strides_for_shape

# def _all_real_strides_old(vm2: View, vm1: View) -> bool:
#   vm1, vm2 = vm1.canonicalize_mask(), vm2.canonicalize_mask()
#   origin, coeffs, strides = _project_view(vm2, vm1)
#   es1 = tuple(e-b for b,e in m1) if (m1 := vm1.mask) else tuple(s if vm1.strides[i]!=0 else 1 for i,s in enumerate(vm1.shape))
#   es2 = tuple(e-b for b,e in m2) if (m2 := vm2.mask) else tuple(s if vm2.strides[i]!=0 else 1 for i,s in enumerate(vm2.shape))
#   print(f"debug2\n{vm1=} {vm2=}\n{es1=}\n{es2=}")
#   print(f"debug3\n{origin=} {strides=}\n{coeffs=}")
#   # trivial cases
#   if (es2 and all(s==0 for s in es2)) or (es1 and all(s==0 for s in es1)): return True
#   if vm1.contiguous and vm1.shape == vm2.shape and vm2.mask is None: return True
#   if vm2.contiguous and vm1.mask is None: return True
#   if m1 is not None and m2 is None:
#     print("RETURN 2")
#     return False
#   if vm1.contiguous: return False
#   if m1 is None and m2 is None and vm1.offset==vm2.offset==0 and vm1.size() == vm2.size():
#     print("RETURN 2")
#     return True

#   return None
  # if m1 is not None or m2 is None:
  #   print("RETURN 3")
  #   return False
  # print("RETURN 5")
  # return True

# def _all_real_strides(vm2: View, vm1: View) -> bool:
#   vm1, vm2 = vm1.canonicalize_mask(), vm2.canonicalize_mask()
#   origin, coeffs, strides = _project_view(vm2, vm1)
#   def apply_mask(view):
#     diff = tuple(e-b for b,e in m) if (m := view.mask) else tuple(s if st!=0 else 1 for s,st in zip(view.shape, view.strides))
#     return tuple(False for _ in view.shape) if (diff and all(d==0 for d in diff)) else tuple(d!=s for d,s in zip(diff, view.shape))
#   valid1, valid2 = map(apply_mask, (vm1, vm2))
#   nonzero_coeffs = tuple(tuple(c.get(i,0)!=0 for i in range(len(vm1.shape))) for c in coeffs)
#   print(f"{coeffs=}")
#   print(f"{nonzero_coeffs=}")
#   print(f"{valid1=}")
#   print(f"{valid2=}")
#   ret = tuple(False for _ in vm1.shape)
#   for coeffi,v in zip(nonzero_coeffs, valid2):
#     if v: ret = tuple((r or c) for r, c in zip(ret, coeffi))
#   print(f"{ret=}")
#   return functools.reduce(operator.and_, ((r or v) for r,v in zip(ret, valid1)))

#     ret = tuple(False for _ in vm1.shape)
#     for coeffi,v in zip(nonzero_coeffs, valid2):
#       if v: ret = tuple((r or c) for r, c in zip(ret, coeffi))
#     print(f"{ret=}")
#     return functools.reduce(operator.and_, ((r or v) for r,v in zip(ret, valid1)))

#     for i,(s,st) in enumerate(zip(vm1.shape, strides)):
#       if abs(st) in stride_masks1:
#         mask = stride_masks1[abs(st)]
#         if mask is not None:
#           b, e = mask
#           if b>=0 and e<=s and (b-e)<s: strides[i] = None

#     for i,(s,st) in enumerate(zip(vm2.shape, vm2.strides)):
#       mask = stride_masks2[abs(st)]
#       if mask is not None and abs(st) in stride_masks1:
#         b, e = mask
#         if b>=0 and e<=s and (b-e)<s:
#           strides[i] = None

# def _all_real_strides(vm2: View, vm1: View) -> bool: return

def _real_strides(vm2: View, vm1: View) -> List[Optional[sint]]:
  # equivalent to `None not in ShapeTracker((vm2, vm1)).real_strides()`
  vm1, vm2 = vm1.canonicalize_mask(), vm2.canonicalize_mask()
  origin, coeffs, strides = _project_view(vm2, vm1)
  es1 = tuple(e-b for b,e in m1) if (m1 := vm1.mask) else tuple(s if vm1.strides[i]!=0 else 1 for i,s in enumerate(vm1.shape))
  es2 = tuple(e-b for b,e in m2) if (m2 := vm2.mask) else tuple(s if vm2.strides[i]!=0 else 1 for i,s in enumerate(vm2.shape))
  if (es2 and all(s==0 for s in es2)) or (es1 and all(s==0 for s in es1)): return cast(List[Optional[sint]], strides)

  stride_map1 = {abs(st):(i,s,st) for i,(s,st) in enumerate(zip(vm1.shape, strides))}
  stride_map2 = {abs(st):(i,s,st) for i,(s,st) in enumerate(zip(vm2.shape, vm2.strides))}

  if DEBUG>=3:
    # def apply_mask(view):
    #   diff = tuple(e-b for b,e in m) if (m := view.mask) else tuple(s if st!=0 else 1 for s,st in zip(view.shape, view.strides))
    #   return diff, tuple(False for _ in view.shape) if (diff and all(d==0 for d in diff)) else tuple(d!=s for d,s in zip(diff, view.shape))
    # valid1, valid2 = map(apply_mask, (vm1, vm2))
    # nonzero_coeffs = tuple(tuple(c.get(i,0)!=0 for i in range(len(vm1.shape))) for c in coeffs)
    print(f"{stride_map1=}\n")
    print(f"{stride_map2=}\n")
    print(f"{coeffs=}")
    # print(f"{nonzero_coeffs=}")
    # print(f"{valid1=}")
    # print(f"{valid2=}")

  for i1,(max1,newst,o) in enumerate(zip(vm1.shape, strides, origin)):
    mask1 = vm1.mask[i1] if vm1.mask else (0,max1)
    if abs(newst) in stride_map2:
      i2, max2, _ = stride_map2[abs(newst)]
      mask2 = vm2.mask[i2] if vm2.mask else (0,max2)
      (b1,e1), (b2,e2) = mask1, mask2
      newb, newe = o+b1, o+e1
      components = coeffs[i2]
      if i1 in components: pass
      if newb<b2 or newe>e2: strides[i1] = None
  return strides

@functools.lru_cache(maxsize=None)
def merge_views(vm2:View, vm1:View) -> Optional[View]:
  # trivial cases
  if vm1.contiguous and vm1.size() == vm2.size():
    if vm1.shape == vm2.shape: return vm2
  if vm2.contiguous: return vm1

  origin1, coeffs1, strides1 = _project_view(vm2, vm1)
  origin2, coeffs2, strides2 = _project_view(vm2, vm2)
  rstrides = (tmp:=ShapeTracker((vm2, vm1))).real_strides()
  strides1 = tuple(strides1)
  rst = tuple(_real_strides(vm2, vm1))
  print(f"\nmerge views\n{tmp=}\n{strides1=}\n{rstrides=}\n{rst=}") # TODO: remove before merge
  # lhs = None not in rstrides
  # rhs = None not in rst
  # if lhs == True:
  # assert lhs == rhs, f"wrong assert! correct answer: {lhs=} {rhs=}"
  # assert rstrides == rst
  if not vm2.mask and vm1.offset == 0 and None not in rstrides:
    return View.create(vm1.shape, cast(Tuple[sint, ...], rstrides), vm2.offset, vm1.mask)
  if vm1.mask:
    for b,e in vm1.mask:
      if not (b < e): return View.create(vm1.shape, (0,) * len(vm1.shape), 0, ((0,0),) * len(vm1.shape))
    return (merged := merge_views(vm2, vm1.shrink(vm1.mask))) and merged.pad(tuple((b,s-e) for (b,e),s in zip(vm1.mask, vm1.shape)))

  # merge dimensions in vm2 if required
  # NOTE: merging too many dimensions can make it difficult to project vm2's mask, hence only combining when required
  idxs: List[Node] = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(vm1.shape)]
  merged_size, merged_term = 1, NumNode(0)
  extents: List[Tuple[sint, Node]] = []
  for term, s, o in zip(reversed(coeffs1), reversed(vm2.shape), reversed(origin1)):
    merged_term += Variable.sum([idxs[d1] * (s1 * merged_size) for d1, s1 in term.items()]) + o * merged_size
    merged_size *= s
    if not (merged_term >= merged_size) and not (merged_term < 0):
      extents.append((merged_size, merged_term))
      merged_size, merged_term = 1, NumNode(0)
  if merged_term: return None

  if (vm2_shape := tuple(s for s,_ in reversed(extents))) != vm2.shape:
    return (reshaped_vm2 := vm2.reshape(vm2_shape)) and merge_views(reshaped_vm2, vm1)

  if vm2.mask:
    # try to project vm2's mask on to vm1
    newb, newe, bad = [0] * len(vm1.shape), list(vm1.shape), False
    for d2, ((b, e), o, (_, t)) in enumerate(zip(vm2.mask, origin1, reversed(extents))):
      if not (t.min < b or t.max >= e): continue
      if not isinstance(o, int) or not isinstance(b, int) or not isinstance(e, int):
        bad = True
        continue
      term = coeffs1[d2]
      if len(term) != 1:
        if not term and newe: newe[0] = 0
        else: bad = True
        continue
      d1, s1 = tuple(term.items())[0]
      if not isinstance(s1, int) or not isinstance(newe[d1], int):
        bad = True
        continue
      newb[d1] = max(newb[d1], math.ceil((b - o if s1 > 0 else e - o - 1) / s1))
      newe[d1] = min(newe[d1], (b - o if s1 < 0 else e - o - 1) // s1 + 1)

    # if any of vm1 was masked off, try again with that mask in place
    for b, e, s in zip(newb, newe, vm1.shape):
      if b != 0 or e != s:
        return merge_views(vm2, View.create(vm1.shape, vm1.strides, vm1.offset, tuple(zip(newb, newe))))
    # otherwise if vm2's mask was violated, then cannot merge
    if bad: return None

  return View.create(vm1.shape, tuple(strides1), sum(o * s for o, s in zip(origin1, vm2.strides)) + vm2.offset)

def _project_position(shape:Tuple[sint, ...], pos:sint) -> List[sint]:
  # projection of position to contig tensor position (with given shape)
  ret= list()
  for stride in strides_for_shape(shape):
    here = pos // stride if stride else 0
    ret.append(here)
    pos -= here * stride
  return ret

def _project_view(vm2:View, vm1:View) -> Tuple[List[sint], List[Dict[int, sint]], List[sint]]:
  # project vm1's offset and strides on to vm2
  origin: List[sint] = _project_position(vm2.shape, vm1.offset)
  if vm1 == vm2: return origin, [{i:1} for i in range(len(origin))], list(vm1.strides)
  coeffs: List[Dict[int, sint]] = [dict() for _ in origin]
  strides: List[sint] = [0] * len(vm1.shape)
  for d1, st in enumerate(vm1.strides):
    if st == 0: continue
    # take a step along d1 (of vm1)
    for d2, (o, idx) in enumerate(zip(origin, _project_position(vm2.shape, vm1.offset + st))):
      if (coeff := idx - o) == 0: continue
      strides[d1] += coeff * vm2.strides[d2]
      coeffs[d2][d1]= coeff
  return origin, coeffs, strides

def _expr_view(view:View, idxs:List[Node], valid:Optional[Node]=None) -> Tuple[Node, Node]:
  assert len(idxs) == len(view.shape), f"need an idx for all dimensions {idxs} vs {view.shape}"
  iexpr: List[Node] = [NumNode(view.offset) if isinstance(view.offset, int) else view.offset]
  vexpr: List[Node] = [valid] if valid is not None else []
  for idx,sh,st,m in zip(idxs, view.shape, view.strides, view.mask if view.mask is not None else [None]*len(view.shape)):
    if sh != 1 and st != 0: iexpr.append(idx*st)
    if m is not None:
      # NOTE: current symbolic behaviour
      # AND(x>=y,x<y) => (((Variable('a', 0, 10)*-1)<((Variable('b', 0, 10)*-1)+NumNode(1))) and (Variable('a', 0, 10)<Variable('b', 0, 10)))
      # AND(x>=0,x<0) => NumNode(0)
      # AND(x>=a,x<a) => (((Variable('v', 0, 10)*-1)<-2) and (Variable('v', 0, 10)<3)), where a>=1 (say 3)
      # AND(x>=a,x<a) => NumNode(0), where a<=-1
      # (bypass symbolic for now!)
      vexpr += [NumNode(0)] if m[0]==m[1] else [idx >= m[0], idx < m[1]]
  return Node.sum(iexpr), Node.ands(vexpr)

@dataclass(frozen=True)
class ShapeTracker:
  views: Tuple[View, ...]
  initsize: Optional[sint] = field(default=None, repr=False, compare=False)

  def __add__(self, st:ShapeTracker) -> ShapeTracker:
    ret = self
    for v in st.simplify().views: ret = ShapeTracker(ret.views + (v,), ret.initsize).simplify() # one view at a time = better simplification
    return ret

  # same as self.canonicalized == other.canonicalized, but faster (no redundant comps)
  def equals(self, other: ShapeTracker):
    # if not isinstance(other, ShapeTracker): raise TypeError(f"expected ShapeTracker, got {type(other)}")
    # return len(self.views)==len(other.views) and all(v1.canonicalize_mask() == v2.canonicalize_mask() for v1,v2 in zip(self.views,other.views))
    if self.views == other.views: return True
    if len(vsa := self.simplify().views) == len(vsb := other.simplify().views):
      if all(va.canonicalize_mask()==vb.canonicalize_mask() or va.minify().canonicalize_mask() == vb.minify().canonicalize_mask() for va, vb in zip(vsa, vsb)): return True  # noqa: E501
    return False

  def invert(self, out_shape:Tuple[sint, ...]) -> Optional[ShapeTracker]:
    ret = tuple(v.invert(s) for v,s in zip(self.views[::-1], [x.shape for x in self.views[::-1][1:]]+[out_shape]))
    return ShapeTracker((cast(Tuple[View, ...], ret))).reshape(out_shape) if all(x is not None for x in ret) else None

  @staticmethod
  def from_shape(shape:Tuple[sint, ...]): return ShapeTracker((View.create(shape),), prod(shape))

  @property
  def canonicalized(self): return ShapeTracker(tuple(v.minify().canonicalize_mask() for v in self.simplify().views))

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def shape(self) -> Tuple[sint, ...]: return self.views[-1].shape

  @property
  def size(self) -> int: return self.views[-1].size()

  def real_size(self) -> int:
    if 0 in self.shape: return 0
    ret = cast(sint, self.expr_idxs()[0].max)   # TODO: this is due to typing issues in symbolic!
    while not isinstance(ret, int): ret = ret.max    # TODO: this is a while loop?!? it should be more clear what max does
    assert isinstance(ret, int), f"ret must be integer, {ret=} isn't"
    return ret+1

  def vars(self) -> Set[Variable]: return set.union(*[v.vars() for v in self.views], set())

  @property
  def var_vals(self) -> Dict[Variable, int]: return merge_dicts([dict([v.unbind()]) for v in self.vars()])

  def unbind(self) -> Tuple[ShapeTracker, Dict[Variable, int]]:
    unbound_views, var_vals = zip(*[v.unbind() for v in self.views])
    return ShapeTracker(tuple(unbound_views), self.initsize), merge_dicts(var_vals)

  # NOTE: if a stride is not always valid, it will be None
  def real_strides(self, ignore_valid=False) -> Tuple[Optional[sint], ...]:
    if len(self.views) == 1 and self.views[-1].mask is None: return self.views[-1].strides
    idxs: List[Node] = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)]
    idx, valid = self.expr_idxs(idxs)
    print(f"\nreal strides\n{idx=}\n{valid=}")
    ret: List[Optional[sint]] = [None] * len(self.shape)
    bad_idx_vars: Set[Variable] = set()
    for this_dim in (idx.nodes if isinstance(idx, SumNode) else [idx]):
      idx_maybe, stride_maybe = (this_dim.a, this_dim.b) if isinstance(this_dim, MulNode) else (this_dim, 1)
      try: ret[idxs.index(idx_maybe)] = stride_maybe
      except ValueError: bad_idx_vars = bad_idx_vars.union(idx_maybe.vars())
    idx_vars, valid_vars = idx.vars(), valid.vars()
    for i,tidx in enumerate(idxs):
      if tidx in bad_idx_vars or (tidx in valid_vars and not ignore_valid): ret[i] = None
      elif tidx not in idx_vars: ret[i] = 0
    return tuple(ret)

  def unit_stride_axes(self, ignore_valid=False) -> List[int]: return [i for i,st in enumerate(self.real_strides(ignore_valid)) if st == 1]

  def expr_idxs(self, idxs:Optional[Iterable[Node]]=None) -> Tuple[Node, Node]:
    idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)] if idxs is None else list(idxs)
    idx, valid = _expr_view(self.views[-1], idxs)
    for view in reversed(self.views[0:-1]):
      if valid.max == 0: return NumNode(-1), valid
      view = view.minify()
      acc, idxs = 1, []
      for d in reversed(view.shape):
        idxs.append((idx//acc)%d)
        acc *= d
      idx, valid = _expr_view(view, idxs[::-1], valid)
    return idx, valid

  def axis_is_masked(self, axis:int) -> bool:
    _, valid = self.expr_idxs()
    return f'idx{axis}' in [v.expr for v in valid.vars()]

  def simplify(self) -> ShapeTracker:
    if len(self.views) >= 2 and (new_view := merge_views(self.views[-2], self.views[-1])) is not None:
      return ShapeTracker(self.views[:-2] + (new_view,)).simplify()
    return self

  # *** under this line are the movement ops ***

  def pad(self, arg: Tuple[Tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].pad(arg), ))
  def shrink(self, arg: Tuple[Tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].shrink(arg), ))
  def expand(self, new_shape: Tuple[sint, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].expand(new_shape), ))
  def permute(self, axis: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].permute(axis), ))
  def stride(self, mul: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].stride(mul), ))
  def reshape(self, new_shape: Tuple[sint, ...]) -> ShapeTracker:
    if getenv("MERGE_VIEW", 1) and (new_view := self.views[-1].reshape(new_shape)) is not None: return ShapeTracker(self.views[0:-1] + (new_view,))
    return ShapeTracker(self.views + (View.create(new_shape), ))
