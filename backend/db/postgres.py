import asyncpg
import os
import json
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=os.getenv("PG_HOST", "localhost"),
            port=int(os.getenv("PG_PORT", 5432)),
            database=os.getenv("PG_DATABASE", "cybershield"),
            user=os.getenv("PG_USER", "postgres"),
            password=os.getenv("PG_PASSWORD", ""),
            min_size=5,
            max_size=20,
        )
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def fetch_one(query: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetch_all(query: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def execute(query: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def notify_channel(channel: str, payload: dict):
    """Send pg NOTIFY to a channel — admin dashboard listens to this"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        payload_str = json.dumps(payload, default=str)
        await conn.execute("SELECT pg_notify($1, $2)", channel, payload_str)
