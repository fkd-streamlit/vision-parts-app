# -*- coding: utf-8 -*-
"""製品マスタ（マニュアル・公式リンク）の読込と UI 表示"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

CATALOG_PATH = Path(__file__).resolve().parent / "product_catalog.json"


@st.cache_data
def load_catalog() -> Dict[str, Any]:
    if not CATALOG_PATH.is_file():
        return {}
    with CATALOG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def get_catalog_entry(class_name: str) -> Optional[Dict[str, Any]]:
    return load_catalog().get(class_name)


def render_manual_section(
    class_name: str,
    *,
    confidence: Optional[float] = None,
    conf_threshold: Optional[float] = None,
    expanded: bool = True,
) -> None:
    """
    判定成功時などにマニュアル・公式リンクのボタンを表示する。
    confidence と conf_threshold が両方ある場合、未満なら案内のみ表示。
    """
    entry = get_catalog_entry(class_name)
    if entry is None:
        st.caption("この製品のマニュアル情報は未登録です。")
        return

    ok_to_link = True
    if confidence is not None and conf_threshold is not None:
        ok_to_link = confidence >= conf_threshold

    with st.expander("📖 マニュアル・技術資料", expanded=expanded and ok_to_link):
        model = entry.get("model_code", class_name)
        series = entry.get("series", "")
        desc = entry.get("description", "")
        st.markdown(f"**{model}** ({series})")
        if desc:
            st.caption(desc)

        if not ok_to_link:
            st.warning(
                f"確信度がしきい値（{conf_threshold*100:.0f}%）未満のため、"
                "誤ったマニュアル表示を避けるためリンクは控えめに表示しています。"
                "再撮影後に確信度が上がると利用できます。"
            )
            return

        manuals: List[Dict[str, str]] = entry.get("manuals") or []
        links: List[Dict[str, str]] = entry.get("links") or []

        if manuals:
            st.markdown("**マニュアル（PDF）**")
            for m in manuals:
                title = m.get("title", "マニュアル")
                url = m.get("url", "")
                note = m.get("note", "")
                if url:
                    st.link_button(f"📄 {title}", url, use_container_width=True)
                if note:
                    st.caption(note)

        if links:
            st.markdown("**関連リンク**")
            cols = st.columns(2)
            for i, link in enumerate(links):
                title = link.get("title", "リンク")
                url = link.get("url", "")
                if not url:
                    continue
                with cols[i % 2]:
                    st.link_button(title, url, use_container_width=True)

        st.caption(
            "※ リンク先はシマノ公式サイトです。PDF・ページはシマノ側の更新により変更される場合があります。"
        )
