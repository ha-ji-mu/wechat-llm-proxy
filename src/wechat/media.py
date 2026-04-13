"""
WeChat CDN media decryption.

Per wechatbot-sdk crypto.py: AES-128-ECB with PKCS7 padding.
  - key: bytes.fromhex(image_item.aeskey)
  - mode: ECB (no IV)
"""
from __future__ import annotations

import base64
import binascii

import httpx
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from loguru import logger


def _is_valid_image(data: bytes) -> bool:
    return (
        data[:3] == b"\xff\xd8\xff"
        or data[:8] == b"\x89PNG\r\n\x1a\n"
        or data[:6] in (b"GIF87a", b"GIF89a")
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    )


def _detect_media_type(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _decode_aes_key(aeskey_hex: str) -> bytes:
    """Convert aeskey hex string to 16-byte key."""
    return binascii.unhexlify(aeskey_hex)


def _decrypt_ecb(data: bytes, key: bytes) -> bytes:
    """AES-128-ECB decrypt with PKCS7 unpadding."""
    cipher = Cipher(algorithms.AES(key), modes.ECB())
    dec = cipher.decryptor()
    padded = dec.update(data) + dec.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    return unpadder.update(padded) + unpadder.finalize()


async def download_image_as_base64(
    url: str, aes_key_hex: str
) -> tuple[str, str]:
    """
    Download and decrypt a WeChat CDN image.

    Returns:
        (base64_data, media_type)  — ready for Anthropic vision content block
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        raw = resp.content

    logger.debug(f"Downloaded {len(raw)} bytes, header={raw[:8].hex()}, mod16={len(raw) % 16}")

    if _is_valid_image(raw):
        logger.debug("CDN returned plaintext image")
        image_bytes = raw
    else:
        key = _decode_aes_key(aes_key_hex)
        image_bytes = _decrypt_ecb(raw, key)
        logger.debug(f"Decrypted to {len(image_bytes)} bytes, header={image_bytes[:8].hex()}")

    media_type = _detect_media_type(image_bytes)
    b64 = base64.standard_b64encode(image_bytes).decode()
    return b64, media_type
