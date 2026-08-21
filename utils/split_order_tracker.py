"""
utils/split_order_tracker.py
...
"""

import asyncio
import json
import os
from typing import Optional

SPLIT_ORDERS_PATH = "./Logs/split_orders_tracker.json"
_split_orders_lock = asyncio.Lock()


def _sync_read() -> dict:
    if not os.path.exists(SPLIT_ORDERS_PATH):
        return {}
    with open(SPLIT_ORDERS_PATH, "r") as f:
        content = f.read().strip()
        return json.loads(content) if content else {}


def _sync_write(state: dict):
    os.makedirs(os.path.dirname(SPLIT_ORDERS_PATH), exist_ok=True)
    with open(SPLIT_ORDERS_PATH, "w") as f:
        json.dump(state, f, indent=4)


async def register_split_order_pair(
    tp1_id: str, sl1_id: str, tp2_id: str, sl2_id: str,
    entry_price: float, ticker: str,
):
    async with _split_orders_lock:
        state = await asyncio.to_thread(_sync_read)

        state[tp1_id] = {
            "type": "TP1_LEG",
            "twin_id": tp2_id,        # twin's TP id -- used by clear_split_order_pair
            "twin_sl_id": sl2_id,     # twin's STOP id -- used by move_twin_to_breakeven
            "entry_price": entry_price, "is_breakeven": False, "ticker": ticker,
        }
        state[tp2_id] = {
            "type": "TP2_LEG",
            "twin_id": tp1_id,
            "twin_sl_id": sl1_id,
            "entry_price": entry_price, "is_breakeven": False, "ticker": ticker,
        }
        # SL legs also need their own tracked records, since a stop fill
        # arrives keyed by the STOP order's own id, not the TP id.
        state[sl1_id] = {
            "type": "SL1_LEG",
            "twin_id": tp2_id,        # twin's TP id -- clear_split_order_pair uses this
            "twin_sl_id": sl2_id,
            "entry_price": entry_price, "is_breakeven": False, "ticker": ticker,
        }
        state[sl2_id] = {
            "type": "SL2_LEG",
            "twin_id": tp1_id,
            "twin_sl_id": sl1_id,
            "entry_price": entry_price, "is_breakeven": False, "ticker": ticker,
        }

        await asyncio.to_thread(_sync_write, state)


async def register_single_order(tp_id: str, sl_id: str, entry_price: float, ticker: str):
    async with _split_orders_lock:
        state = await asyncio.to_thread(_sync_read)
        state[tp_id] = {
            "type": "TP_SINGLE",
            "twin_id": "NONE", "twin_sl_id": "NONE",
            "entry_price": entry_price, "is_breakeven": False, "ticker": ticker,
        }
        state[sl_id] = {
            "type": "SL_SINGLE",
            "twin_id": "NONE", "twin_sl_id": "NONE",
            "entry_price": entry_price, "is_breakeven": False, "ticker": ticker,
        }
        await asyncio.to_thread(_sync_write, state)


async def get_split_order_leg(order_id: str) -> Optional[dict]:
    """Returns the leg_info dict for order_id, or None if it isn't tracked."""
    async with _split_orders_lock:
        state = await asyncio.to_thread(_sync_read)
        return state.get(order_id)


async def find_leg_by_ticker(ticker: str) -> Optional[tuple]:
    async with _split_orders_lock:
        state = await asyncio.to_thread(_sync_read)
        for order_id, leg_info in state.items():
            if leg_info.get("ticker") == ticker:
                return order_id, leg_info
    return None


async def set_breakeven_flag(order_id: str):
    """Marks a leg as moved to breakeven (called from move_twin_to_breakeven)."""
    async with _split_orders_lock:
        state = await asyncio.to_thread(_sync_read)
        if order_id in state:
            state[order_id]["is_breakeven"] = True
            await asyncio.to_thread(_sync_write, state)


async def clear_split_order_pair(*order_ids: str):
    async with _split_orders_lock:
        state = await asyncio.to_thread(_sync_read)
        for oid in order_ids:
            if oid and oid != "NONE":
                state.pop(oid, None)
        await asyncio.to_thread(_sync_write, state)