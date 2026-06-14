"""Memory reading and writing operations for IDA Pro MCP.

This module provides batch operations for reading and writing memory at various
granularities (bytes, integers, strings) and patching binary data.
"""

import re

from typing import Annotated, NotRequired, TypedDict
import ida_bytes
import idaapi

from .rpc import tool
from .sync import idasync
from .utils import (
    IntRead,
    IntWrite,
    MemoryPatch,
    MemoryRead,
    normalize_list_input,
    parse_address,
    read_bytes_bss_safe,
    read_int_bss_safe,
)


class BytesReadResult(TypedDict):
    addr: str | None
    data: str | None
    error: NotRequired[str]


class IntReadResult(TypedDict):
    addr: str
    ty: str
    value: int | None
    error: NotRequired[str]


class StringReadResult(TypedDict):
    addr: str
    value: str | None
    error: NotRequired[str]


class GlobalValueResult(TypedDict):
    query: str
    value: str | None
    error: NotRequired[str]


class PatchResult(TypedDict):
    addr: str | None
    size: int
    error: NotRequired[str]


class IntWriteResult(TypedDict):
    addr: str
    ty: str
    value: str | None
    error: NotRequired[str]


class PatchedBlock(TypedDict):
    start_addr: str
    end_addr: str
    size: int
    original: str
    patched: str


# ============================================================================
# Memory Reading Operations
# ============================================================================


@tool
@idasync
def get_bytes(regions: list[MemoryRead] | MemoryRead) -> list[BytesReadResult]:
    """Read bytes from memory addresses"""
    if isinstance(regions, dict):
        regions = [regions]

    results = []
    for item in regions:
        addr = item.get("addr", "")
        size = item.get("size", 0)

        try:
            ea = parse_address(addr)
            raw = read_bytes_bss_safe(ea, size)
            data = " ".join(f"{x:#02x}" for x in raw)
            results.append({"addr": addr, "data": data})
        except Exception as e:
            results.append({"addr": addr, "data": None, "error": str(e)})

    return results


_INT_CLASS_RE = re.compile(r"^(?P<sign>[iu])(?P<bits>8|16|32|64)(?P<endian>le|be)?$")


def _parse_int_class(text: str) -> tuple[int, bool, str, str]:
    if not text:
        raise ValueError("Missing integer class")

    cleaned = text.strip().lower()
    match = _INT_CLASS_RE.match(cleaned)
    if not match:
        raise ValueError(f"Invalid integer class: {text}")

    bits = int(match.group("bits"))
    signed = match.group("sign") == "i"
    endian = match.group("endian") or "le"
    byte_order = "little" if endian == "le" else "big"
    normalized = f"{'i' if signed else 'u'}{bits}{endian}"
    return bits, signed, byte_order, normalized


def _parse_int_value(text: str, signed: bool, bits: int) -> int:
    if text is None:
        raise ValueError("Missing integer value")

    value_text = str(text).strip()
    try:
        value = int(value_text, 0)
    except ValueError:
        raise ValueError(f"Invalid integer value: {text}")

    if not signed and value < 0:
        raise ValueError(f"Negative value not allowed for u{bits}")

    return value


@tool
@idasync
def get_int(
    queries: Annotated[
        list[IntRead] | IntRead,
        "Integer read requests (ty, addr). ty: i8/u64/i16le/i16be/etc",
    ],
) -> list[IntReadResult]:
    """Read integer values from memory addresses"""
    if isinstance(queries, dict):
        queries = [queries]

    results = []
    for item in queries:
        addr = item.get("addr", "")
        ty = item.get("ty", "")

        try:
            bits, signed, byte_order, normalized = _parse_int_class(ty)
            ea = parse_address(addr)
            size = bits // 8
            data = read_bytes_bss_safe(ea, size)
            if len(data) != size:
                raise ValueError(f"Failed to read {size} bytes at {addr}")

            value = int.from_bytes(data, byte_order, signed=signed)
            results.append(
                {"addr": addr, "ty": normalized, "value": value}
            )
        except Exception as e:
            results.append({"addr": addr, "ty": ty, "value": None, "error": str(e)})

    return results


@tool
@idasync
def get_string(
    addrs: Annotated[list[str] | str, "Addresses to read strings from"],
) -> list[StringReadResult]:
    """Read strings from memory addresses"""
    addrs = normalize_list_input(addrs)
    results = []

    for addr in addrs:
        try:
            ea = parse_address(addr)
            raw = idaapi.get_strlit_contents(ea, -1, 0)
            if not raw:
                results.append(
                    {"addr": addr, "value": None, "error": "No string at address"}
                )
                continue
            value = raw.decode("utf-8", errors="replace")
            results.append({"addr": addr, "value": value})
        except Exception as e:
            results.append({"addr": addr, "value": None, "error": str(e)})

    return results


def get_global_variable_value_internal(ea: int) -> str:
    import ida_typeinf
    import ida_nalt
    import ida_bytes
    from .sync import IDAError

    tif = ida_typeinf.tinfo_t()
    if not ida_nalt.get_tinfo(tif, ea):
        if not ida_bytes.has_any_name(ea):
            raise IDAError(f"Failed to get type information for variable at {ea:#x}")

        size = ida_bytes.get_item_size(ea)
        if size == 0:
            raise IDAError(f"Failed to get type information for variable at {ea:#x}")
    else:
        size = tif.get_size()

    if size == 0 and tif.is_array() and tif.get_array_element().is_decl_char():
        raw = idaapi.get_strlit_contents(ea, -1, 0)
        if not raw:
            return '""'
        return_string = raw.decode("utf-8", errors="replace").strip()
        return f'"{return_string}"'

    if size in (1, 2, 4, 8):
        return hex(read_int_bss_safe(ea, size))
    return " ".join(hex(b) for b in read_bytes_bss_safe(ea, size))


@tool
@idasync
def get_global_value(
    queries: Annotated[
        list[str] | str, "Global variable addresses or names to read values from"
    ],
) -> list[GlobalValueResult]:
    """Read global variable values by address or symbol name."""
    from .utils import looks_like_address

    queries = normalize_list_input(queries)
    results = []

    for query in queries:
        try:
            ea = idaapi.BADADDR

            # Try as address first if it looks like one
            if looks_like_address(query):
                try:
                    ea = parse_address(query)
                except Exception:
                    ea = idaapi.BADADDR

            # Fall back to name lookup
            if ea == idaapi.BADADDR:
                ea = idaapi.get_name_ea(idaapi.BADADDR, query)

            if ea == idaapi.BADADDR:
                results.append({"query": query, "value": None, "error": "Not found"})
                continue

            value = get_global_variable_value_internal(ea)
            results.append({"query": query, "value": value})
        except Exception as e:
            results.append({"query": query, "value": None, "error": str(e)})

    return results


# ============================================================================
# Batch Data Operations
# ============================================================================


@tool
@idasync
def patch(patches: list[MemoryPatch] | MemoryPatch) -> list[PatchResult]:
    """Patch bytes at memory addresses with hex data"""
    if isinstance(patches, dict):
        patches = [patches]

    results = []

    for patch in patches:
        try:
            ea = parse_address(patch["addr"])
            data = bytes.fromhex(patch["data"])

            if not ida_bytes.is_mapped(ea):
                raise ValueError(f"Address not mapped: {patch['addr']}")

            ida_bytes.patch_bytes(ea, data)
            results.append(
                {"addr": patch["addr"], "size": len(data)}
            )

        except Exception as e:
            results.append({"addr": patch.get("addr"), "size": 0, "error": str(e)})

    return results


@tool
@idasync
def put_int(
    items: Annotated[
        list[IntWrite] | IntWrite,
        "Integer write requests (ty, addr, value). value is a string; supports 0x.. and negatives",
    ],
) -> list[IntWriteResult]:
    """Write integer values to memory addresses"""
    if isinstance(items, dict):
        items = [items]

    results = []
    for item in items:
        addr = item.get("addr", "")
        ty = item.get("ty", "")
        value_text = item.get("value")

        try:
            bits, signed, byte_order, normalized = _parse_int_class(ty)
            value = _parse_int_value(value_text, signed, bits)
            size = bits // 8
            try:
                data = value.to_bytes(size, byte_order, signed=signed)
            except OverflowError:
                raise ValueError(f"Value {value_text} does not fit in {normalized}")

            ea = parse_address(addr)
            if not ida_bytes.is_mapped(ea):
                raise ValueError(f"Address not mapped: {addr}")
            ida_bytes.patch_bytes(ea, data)
            results.append(
                {
                    "addr": addr,
                    "ty": normalized,
                    "value": str(value_text),
                }
            )
        except Exception as e:
            results.append(
                {
                    "addr": addr,
                    "ty": ty,
                    "value": str(value_text) if value_text is not None else None,
                    "error": str(e),
                }
            )

    return results


@tool
@idasync
def get_patched_bytes(
    start_addr: Annotated[
        str | None,
        "Optional starting linear address (hex or decimal) to search from. Defaults to min_ea.",
    ] = None,
    end_addr: Annotated[
        str | None,
        "Optional ending linear address (hex or decimal) to search to. Defaults to max_ea.",
    ] = None,
) -> list[PatchedBlock]:
    """Retrieve list of patched (modified) bytes grouped into contiguous blocks."""
    from . import compat

    try:
        ea_start = parse_address(start_addr) if (start_addr and start_addr.strip()) else compat.inf_get_min_ea()
    except Exception:
        raise ValueError(f"Invalid start address: {start_addr}")

    try:
        ea_end = parse_address(end_addr) if (end_addr and end_addr.strip()) else compat.inf_get_max_ea()
    except Exception:
        raise ValueError(f"Invalid end address: {end_addr}")

    patched_bytes = []

    def callback(ea, fpos, org_val, patch_val):
        patched_bytes.append((ea, fpos, org_val, patch_val))
        return 0

    ida_bytes.visit_patched_bytes(ea_start, ea_end, callback)

    blocks: list[PatchedBlock] = []
    current_block = None

    patched_bytes.sort(key=lambda x: x[0])

    for ea, fpos, org_val, patch_val in patched_bytes:
        if current_block is None:
            current_block = {
                "start_addr": f"{ea:#x}",
                "end_addr": f"{ea + 1:#x}",
                "size": 1,
                "original_list": [org_val],
                "patched_list": [patch_val],
                "last_ea": ea,
            }
        elif ea == current_block["last_ea"] + 1:
            current_block["end_addr"] = f"{ea + 1:#x}"
            current_block["size"] += 1
            current_block["original_list"].append(org_val)
            current_block["patched_list"].append(patch_val)
            current_block["last_ea"] = ea
        else:
            blocks.append({
                "start_addr": current_block["start_addr"],
                "end_addr": current_block["end_addr"],
                "size": current_block["size"],
                "original": " ".join(f"{b:02x}" for b in current_block["original_list"]),
                "patched": " ".join(f"{b:02x}" for b in current_block["patched_list"]),
            })
            current_block = {
                "start_addr": f"{ea:#x}",
                "end_addr": f"{ea + 1:#x}",
                "size": 1,
                "original_list": [org_val],
                "patched_list": [patch_val],
                "last_ea": ea,
            }

    if current_block is not None:
        blocks.append({
            "start_addr": current_block["start_addr"],
            "end_addr": current_block["end_addr"],
            "size": current_block["size"],
            "original": " ".join(f"{b:02x}" for b in current_block["original_list"]),
            "patched": " ".join(f"{b:02x}" for b in current_block["patched_list"]),
        })

    return blocks
