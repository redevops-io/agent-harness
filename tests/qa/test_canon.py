# SPDX-License-Identifier: AGPL-3.0-or-later
"""canon-v1 against the golden vectors (97 canonical-hash vectors, 3 refusals), plus edge cases and mutations."""
import hashlib
import json
import sys

import pytest

from agent_harness.qa import canon
from agent_harness.qa.errors import CanonicalizationRefused

from ._helpers import GOLDEN, GOLDEN_SHA256, CANON_VECTOR_COUNT, load_mutant, load_vectors

DATA = load_vectors()
VECTORS = DATA["vectors"]
REFUSALS = DATA["refusals"]


def test_golden_files_are_the_frozen_if1_files():
    # the golden data is never silently modified to suit the harness: the copies must be byte-identical to the documented hashes
    for name, expected in GOLDEN_SHA256.items():
        assert hashlib.sha256((GOLDEN / name).read_bytes()).hexdigest() == expected, name


def test_vector_file_shape_and_count():
    assert DATA["canonicalization"] == "rcv1" and DATA["contract"] == "qa/v1"
    assert DATA["vector_set"] == "runtime-contracts/qa/canonical-vectors"
    assert len(VECTORS) == CANON_VECTOR_COUNT == 97
    ids = [v["id"] for v in VECTORS]
    assert len(set(ids)) == len(ids)
    assert len(REFUSALS) == 3


@pytest.mark.parametrize("vec", VECTORS, ids=[v["id"] for v in VECTORS])
def test_vector_is_reproduced_byte_for_byte(vec):
    text = canon.canonical_text(vec["input"])
    data = canon.canonical_bytes(vec["input"])
    assert text == vec["canonical_json"]
    assert data == vec["canonical_json"].encode("utf-8")           # UTF-8, no \u escaping of non-ASCII
    assert hashlib.sha256(data).hexdigest() == vec["sha256_hex"]
    assert canon.digest_hex(vec["input"]) == vec["sha256_hex"]
    assert canon.canon_hash(vec["input"]) == "canon-v1:" + vec["sha256_hex"]
    assert vec["rcv1"] == "rcv1:" + vec["sha256_hex"]
    assert canon.to_contract_ref(canon.canon_hash(vec["input"])) == vec["rcv1"]      # prefix mapping
    assert canon.from_contract_ref(vec["rcv1"]) == canon.canon_hash(vec["input"])
    assert canon.same_digest(canon.canon_hash(vec["input"]), vec["rcv1"])


@pytest.mark.parametrize("ref", REFUSALS, ids=[r["id"] for r in REFUSALS])
def test_refusal_vector_is_refused(ref):
    assert ref["expect"] == "refused"
    with pytest.raises(CanonicalizationRefused) as err:
        canon.canonical_bytes(ref["input"])
    assert err.value.code == "CANON_REFUSED"
    with pytest.raises(CanonicalizationRefused):
        canon.canon_hash(ref["input"])


def test_vectors_cover_the_published_rules():
    ids = {v["id"] for v in VECTORS}
    assert {"key_order", "null_dropped_in_map", "null_kept_in_list", "nfc_normalization", "nfc_in_keys",
            "string_escapes", "non_ascii_is_not_escaped", "key_order_is_code_point_order",
            "decimal_string_stays_a_string", "integers"} <= ids
    kinds = {i.split(":")[0] if ":" in i else "base" for i in ids}
    assert kinds == {"base", "qa", "qa-value", "qa-assertion-set"}


# --- edge cases the vectors do not state (fail-closed readings) -----------------------------------

def test_equal_after_nfc_keys_are_refused_not_overwritten():
    with pytest.raises(CanonicalizationRefused):
        canon.canonical_text({"é": 1, "é": 2})


@pytest.mark.parametrize("bad", [{1: "x"}, {None: "x"}, {("a",): 1}])
def test_non_string_keys_are_refused(bad):
    with pytest.raises(CanonicalizationRefused):
        canon.canonical_text(bad)


@pytest.mark.parametrize("bad", [b"bytes", {1, 2}, object(), 1 + 2j, float("nan"), float("inf"), -0.0, 2.5])
def test_unsupported_values_are_refused(bad):
    with pytest.raises(CanonicalizationRefused):
        canon.canonical_text({"a": bad})
    with pytest.raises(CanonicalizationRefused):
        canon.canonical_text([bad])


def test_scalar_subclasses_are_refused_exact_types_only():
    import enum

    class N(enum.IntEnum):
        A = 1

    class S(str):
        pass

    with pytest.raises(CanonicalizationRefused):
        canon.canonical_text({"a": N.A})
    with pytest.raises(CanonicalizationRefused):
        canon.canonical_text({"a": S("x")})


def test_lone_surrogate_is_refused():
    with pytest.raises(CanonicalizationRefused):
        canon.canonical_text({"a": "\ud800"})
    with pytest.raises(CanonicalizationRefused):
        canon.canonical_text({"\ud800": 1})


def test_depth_limit_refuses_instead_of_recursing():
    deep = cyc = []
    for _ in range(canon.MAX_DEPTH + 5):
        nxt = []
        cyc.append(nxt)
        cyc = nxt
    with pytest.raises(CanonicalizationRefused):
        canon.canonical_text(deep)
    loop = {}
    loop["self"] = loop
    with pytest.raises(CanonicalizationRefused):
        canon.canonical_text(loop)
    ok = cur = []
    for _ in range(canon.MAX_DEPTH - 1):
        nxt = []
        cur.append(nxt)
        cur = nxt
    canon.canonical_text(ok)


def test_integers_of_any_size_and_negative_zero_free_output():
    assert canon.canonical_text([0, -1, 2 ** 80, True, False, None]) == "[0,-1," + str(2 ** 80) + ",true,false,null]"


def test_top_level_null_and_empty_map_drops_everything():
    assert canon.canonical_text(None) == "null"
    assert canon.canonical_text({"a": None, "b": None}) == "{}"
    assert canon.canonical_text({"a": {"b": None}}) == '{"a":{}}'


def test_control_characters_match_standard_json_escaping():
    for code in list(range(0x20)) + [0x22, 0x5C, 0x2F, 0x7F, 0x2028, 0xFFFF]:
        ch = chr(code)
        assert canon.canonical_text(ch) == json.dumps(ch, ensure_ascii=False), hex(code)


def test_key_order_is_independent_of_insertion_order():
    for v in VECTORS:
        assert canon.canonical_text(_reversed(v["input"])) == v["canonical_json"], v["id"]


def test_tuples_and_lists_and_mappings_are_interchangeable():
    from agent_harness.qa.records import FrozenMap
    assert canon.canonical_text({"a": (1, 2), "b": FrozenMap({"y": 1, "x": 2})}) == '{"a":[1,2],"b":{"x":2,"y":1}}'


def test_prefix_mapping_is_strict():
    h = "a" * 64
    assert canon.to_contract_ref("canon-v1:" + h) == "rcv1:" + h
    assert canon.from_contract_ref("rcv1:" + h) == "canon-v1:" + h
    assert canon.same_digest("canon-v1:" + h, "rcv1:" + h)
    assert not canon.same_digest("canon-v1:" + h, "rcv1:" + "b" * 64)
    for bad in ("", "sha256:" + h, "rcv1:" + h.upper(), "rcv1:" + h[:-1], "rcv1:" + h + "0", "canon-v1:" + h[:16]):
        with pytest.raises(CanonicalizationRefused):
            canon.digest_of(bad)
    with pytest.raises(CanonicalizationRefused):
        canon.to_contract_ref("rcv1:" + h)
    with pytest.raises(CanonicalizationRefused):
        canon.from_contract_ref("canon-v1:" + h)


# --- mutation tests: removing a rule must make this suite fail --------------------------------------------------

def _reversed(value):
    if isinstance(value, dict):
        return {k: _reversed(value[k]) for k in reversed(list(value))}
    if isinstance(value, list):
        return [_reversed(x) for x in value]
    return value


def failing_checks(mod):
    """Names of the checks of this file that module ``mod`` (real or mutated canon) fails."""
    failed = set()
    for v in VECTORS:
        try:
            ok = mod.canonical_text(v["input"]) == v["canonical_json"] and mod.digest_hex(v["input"]) == v["sha256_hex"]
            ok = ok and mod.canonical_text(_reversed(v["input"])) == v["canonical_json"]
        except Exception:
            ok = False
        if not ok:
            failed.add(v["id"])
    for r in REFUSALS:
        try:
            mod.canonical_text(r["input"])
            failed.add("refusal:" + r["id"])
        except CanonicalizationRefused:
            pass
    return failed


def _refuses(mod, value):
    try:
        mod.canonical_text(value)
    except CanonicalizationRefused:
        return True
    except Exception:
        return False
    return False


@pytest.fixture
def mutant():
    made = []

    def make(replacements):
        mod = load_mutant("canon.py", replacements)
        made.append(mod.__name__)
        return mod

    yield make
    for name in made:
        sys.modules.pop(name, None)


def test_the_unmutated_module_fails_nothing(mutant):
    assert failing_checks(canon) == set()
    assert failing_checks(mutant([])) == set()


VECTOR_MUTATIONS = {
    "floats_accepted": ([('raise CanonicalizationRefused("a float has no canonical form; carry decimals as strings")',
                          "out.append(repr(value))")], {"refusal:float_refused", "refusal:float_in_list_refused"}),
    "no_nfc": ([('return unicodedata.normalize("NFC", text)', "return text")], {"nfc_normalization", "nfc_in_keys"}),
    "keys_not_sorted": ([("for nkey in sorted(items):                      # str ordering is by Unicode code point",
                          "for nkey in items:")], {"key_order", "nested_key_order"}),
    "keys_sorted_by_utf16": ([("for nkey in sorted(items):                      # str ordering is by Unicode code point",
                               'for nkey in sorted(items, key=lambda k: k.encode("utf-16-be")):')],
                             {"key_order_is_code_point_order"}),
    "null_values_kept": ([("            if item is None:                            # absent equals null: a null map value is dropped\n"
                           "                continue\n", "")], {"null_dropped_in_map"}),
    "non_ascii_escaped": ([("        else:\n            out.append(ch)\n    out.append('\"')",
                            "        else:\n            out.append(ch if ord(ch) < 128 else \"\\\\u%04x\" % ord(ch))\n    out.append('\"')")],
                          {"non_ascii_is_not_escaped"}),
    "spaced_separators": ([("            if not first:\n                out.append(\",\")",
                            "            if not first:\n                out.append(\", \")")], {"compact_separators"}),
    "slash_and_del_escaped": ([('elif ch < " ":', 'elif ch < " " or ch in "/\\x7f":')], {"string_escapes"}),
    "booleans_as_numbers": ([('out.append("true" if value else "false")', 'out.append("1" if value else "0")')],
                            {"booleans"}),
}


@pytest.mark.parametrize("name", sorted(VECTOR_MUTATIONS))
def test_removing_a_rule_fails_the_vectors(mutant, name):
    replacements, expected = VECTOR_MUTATIONS[name]
    failed = failing_checks(mutant(replacements))
    assert failed, f"mutation {name} survived: nothing in the suite noticed"
    assert expected <= failed, (name, sorted(expected - failed))


EDGE_MUTATIONS = {
    "duplicate_nfc_keys_allowed": ([('            if nkey in items:\n                raise CanonicalizationRefused('
                                     'f"two map keys are equal after NFC normalization: {nkey!r}")\n', "")],
                                   {"é": 1, "é": 2}),
    "non_string_keys_allowed": ([('            if type(key) is not str:\n                raise CanonicalizationRefused('
                                  'f"map keys must be strings, got {type(key).__name__}")\n', "")], {1: "x"}),
    "depth_unbounded": ([("    if depth > MAX_DEPTH:\n        raise CanonicalizationRefused("
                          'f"nesting deeper than {MAX_DEPTH} levels is refused")\n', "")],
                        None),
    "lone_surrogate_allowed": ([("    except UnicodeEncodeError as exc:\n        raise CanonicalizationRefused("
                                 'f"{what} is not valid Unicode text (lone surrogate)") from exc\n',
                                 "    except UnicodeEncodeError as exc:\n        raise\n")], {"a": "\ud800"}),
}


def _deep(n):
    root = cur = []
    for _ in range(n):
        nxt = []
        cur.append(nxt)
        cur = nxt
    return root


@pytest.mark.parametrize("name", sorted(EDGE_MUTATIONS))
def test_removing_an_edge_guard_is_noticed(mutant, name):
    replacements, probe = EDGE_MUTATIONS[name]
    probe = _deep(canon.MAX_DEPTH + 40) if probe is None else probe
    assert _refuses(canon, probe), "the real module must refuse the probe"
    assert not _refuses(mutant(replacements), probe), f"mutation {name} survived"
