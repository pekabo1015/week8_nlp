import html
import re
from typing import Dict, List, Optional, Tuple

import spacy
import streamlit as st
from streamlit_echarts import st_echarts


DEFAULT_TEXT = "马云在阿里巴巴工作，并与Elon Musk在上海会面。"

ENTITY_COLORS = {
    "PER": "#ffd8a8",
    "ORG": "#d3f9d8",
    "LOC": "#c5f6fa",
}

ENTITY_LABEL_MAP = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
}

GRAPH_STYLE = {
    "PER": {"color": "#f4a261", "size": 68},
    "ORG": {"color": "#2a9d8f", "size": 76},
    "LOC": {"color": "#457b9d", "size": 62},
    "UNK": {"color": "#6c757d", "size": 56},
}


def _add_entity(entities: List[Dict], text: str, start: int, end: int, label: str) -> None:
    if start < 0 or end <= start or end > len(text):
        return
    span_text = text[start:end].strip()
    if not span_text:
        return
    entities.append({"start": start, "end": end, "text": span_text, "label": label})


def _resolve_overlaps(entities: List[Dict]) -> List[Dict]:
    # 优先保留更长片段，避免短片段覆盖长实体
    entities.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    filtered: List[Dict] = []
    for ent in entities:
        has_overlap = False
        for kept in filtered:
            if not (ent["end"] <= kept["start"] or ent["start"] >= kept["end"]):
                has_overlap = True
                break
        if not has_overlap:
            filtered.append(ent)
    filtered.sort(key=lambda x: x["start"])
    return filtered


def rule_extract_entities(text: str) -> List[Dict]:
    """
    使用简单词典匹配进行 Mock 实体抽取。
    后续可替换为 spaCy 或大模型 API。
    """
    lexicon = [
        ("马云", "PER"),
        ("Elon Musk", "PER"),
        ("阿里巴巴", "ORG"),
        ("上海", "LOC"),
        ("微软", "ORG"),
        ("北京", "LOC"),
        ("张伟", "PER"),
        ("Google", "ORG"),
        ("New York", "LOC"),
        ("腾讯", "ORG"),
        ("字节跳动", "ORG"),
        ("杭州", "LOC"),
        ("深圳", "LOC"),
    ]
    entities: List[Dict] = []

    for phrase, label in lexicon:
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx == -1:
                break
            _add_entity(entities, text, idx, idx + len(phrase), label)
            start = idx + len(phrase)

    # 规则1：英文人名（两个及以上首字母大写词）
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
        _add_entity(entities, text, m.start(1), m.end(1), "PER")

    # 规则2：中英机构名（常见机构后缀）
    org_pattern = (
        r"([\u4e00-\u9fa5A-Za-z]{2,30}"
        r"(?:公司|集团|大学|学院|研究院|实验室|银行|委员会|政府|医院|法院|检察院|中心|局|部|厅|"
        r"Inc\.?|Corp\.?|Corporation|University|Institute|Lab|Labs|Committee|Bank))"
    )
    for m in re.finditer(org_pattern, text):
        _add_entity(entities, text, m.start(1), m.end(1), "ORG")

    # 规则3：地点（中文地名后缀 / 英文 in|at 后专有地名）
    cn_loc_pattern = r"([\u4e00-\u9fa5]{2,12}(?:省|市|区|县|镇|乡|村|国|州))"
    for m in re.finditer(cn_loc_pattern, text):
        _add_entity(entities, text, m.start(1), m.end(1), "LOC")

    en_loc_pattern = r"\b(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
    for m in re.finditer(en_loc_pattern, text):
        _add_entity(entities, text, m.start(1), m.end(1), "LOC")

    # 规则4：中文称谓前的人名（如“张伟先生”）
    cn_per_pattern = r"([\u4e00-\u9fa5]{2,3})(?:先生|女士|老师|教授|博士)"
    for m in re.finditer(cn_per_pattern, text):
        _add_entity(entities, text, m.start(1), m.end(1), "PER")

    return _resolve_overlaps(entities)


def _map_spacy_label_to_demo_label(label: str) -> Optional[str]:
    label_upper = label.upper()
    if label_upper in {"PERSON", "PER"}:
        return "PER"
    if label_upper in {"ORG", "ORGANIZATION"}:
        return "ORG"
    if label_upper in {"GPE", "LOC", "LOCATION", "FAC"}:
        return "LOC"
    return None


@st.cache_resource(show_spinner=False)
def load_spacy_model():
    """
    优先英文小模型，再尝试中文小模型。
    用户可自行在环境中下载模型。
    """
    errors = []
    for model_name in ("en_core_web_sm", "zh_core_web_sm"):
        try:
            return spacy.load(model_name), model_name, None
        except Exception as exc:  # pragma: no cover
            errors.append(f"{model_name}: {exc}")
    return None, None, " | ".join(errors)


def spacy_extract_entities(text: str) -> List[Dict]:
    nlp, _, _ = load_spacy_model()
    rule_entities = rule_extract_entities(text)
    if nlp is None:
        return rule_entities

    doc = nlp(text)
    entities: List[Dict] = []
    for ent in doc.ents:
        mapped = _map_spacy_label_to_demo_label(ent.label_)
        if mapped is None:
            continue
        _add_entity(entities, text, ent.start_char, ent.end_char, mapped)
    # 融合 spaCy 与规则结果，提升对中英混合输入的稳定识别能力。
    entities.extend(rule_entities)
    return _resolve_overlaps(entities)


def resolve_pronoun_coreference(text: str, entities: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    轻量级指代消解：
    - he/him/his/she/her/hers -> 最近的 PER
    - it/its -> 最近的 ORG
    """
    coref_links: List[Dict] = []
    augmented_entities = entities.copy()

    person_entities = sorted([e for e in entities if e["label"] == "PER"], key=lambda x: x["start"])
    org_entities = sorted([e for e in entities if e["label"] == "ORG"], key=lambda x: x["start"])

    pronoun_rules = [
        (r"\b(he|him|his|she|her|hers)\b", "PER"),
        (r"\b(it|its)\b", "ORG"),
    ]

    def find_antecedent(candidates: List[Dict], mention_start: int) -> Optional[Dict]:
        before = [c for c in candidates if c["start"] < mention_start]
        if before:
            return before[-1]
        return candidates[-1] if candidates else None

    for pattern, label in pronoun_rules:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            mention = m.group(1)
            candidates = person_entities if label == "PER" else org_entities
            antecedent = find_antecedent(candidates, m.start(1))
            if antecedent is None:
                continue

            pronoun_ent = {
                "start": m.start(1),
                "end": m.end(1),
                "text": mention,
                "label": label,
                "is_pronoun": True,
                "resolved_to": antecedent["text"],
            }
            augmented_entities.append(pronoun_ent)
            coref_links.append(
                {
                    "mention": mention,
                    "mention_start": m.start(1),
                    "mention_end": m.end(1),
                    "label": label,
                    "antecedent": antecedent["text"],
                }
            )

    return _resolve_overlaps(augmented_entities), coref_links


def extract_relations(text: str, entities: List[Dict], coref_links: List[Dict]) -> List[Dict]:
    relations: List[Dict] = []
    persons = [e for e in entities if e["label"] == "PER"]
    orgs = [e for e in entities if e["label"] == "ORG"]
    locs = [e for e in entities if e["label"] == "LOC"]

    def add_relation(source: str, target: str, relation: str) -> None:
        for item in relations:
            if item["source"] == source and item["target"] == target and item["relation"] == relation:
                return
        relations.append({"source": source, "target": target, "relation": relation})

    # 1) FOUNDER_OF
    founder_en = re.finditer(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:founded|co-founded|is the founder of)\s+([A-Z][A-Za-z0-9&\-\s]+)",
        text,
        re.IGNORECASE,
    )
    for m in founder_en:
        add_relation(m.group(1).strip(), m.group(2).strip(), "FOUNDER_OF")

    # 2) 中文“创立/创办”关系
    for m in re.finditer(r"([\u4e00-\u9fa5]{2,6})(?:创立|创办|创建)了?([\u4e00-\u9fa5A-Za-z]{2,30})", text):
        add_relation(m.group(1).strip(), m.group(2).strip(), "FOUNDER_OF")

    # 3) WORKS_FOR
    for m in re.finditer(r"([\u4e00-\u9fa5]{2,6}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)在([\u4e00-\u9fa5A-Za-z]{2,30})工作", text):
        add_relation(m.group(1).strip(), m.group(2).strip(), "WORKS_FOR")
    for m in re.finditer(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:works at|works for|worked at)\s+([A-Z][A-Za-z0-9&\-\s]+)",
        text,
        re.IGNORECASE,
    ):
        add_relation(m.group(1).strip(), m.group(2).strip(), "WORKS_FOR")

    # 4) MEET_WITH / MEET_AT
    meet_pattern = re.finditer(
        r"([\u4e00-\u9fa5]{2,6}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)与([\u4e00-\u9fa5]{2,6}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)在([\u4e00-\u9fa5A-Za-z\s]{2,30})会面",
        text,
    )
    for m in meet_pattern:
        p1, p2, loc = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        add_relation(p1, p2, "MEET_WITH")
        add_relation(p1, loc, "MEET_AT")

    # 5) CEO_OF
    for m in re.finditer(
        r"([\u4e00-\u9fa5]{2,6}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        r"(?:是|担任|出任|成为|担任了)?(?:CEO|首席执行官)"
        r"(?:of|于|在)?([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})",
        text,
        re.IGNORECASE,
    ):
        add_relation(m.group(1).strip(), m.group(2).strip(), "CEO_OF")

    # 6) LOCATED_IN / HEADQUARTER_IN
    for m in re.finditer(r"([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})(?:位于|坐落于)([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})", text):
        add_relation(m.group(1).strip(), m.group(2).strip(), "LOCATED_IN")
    for m in re.finditer(
        r"([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})(?:总部位于|总部在|is headquartered in)\s*([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})",
        text,
        re.IGNORECASE,
    ):
        add_relation(m.group(1).strip(), m.group(2).strip(), "HEADQUARTER_IN")

    # 7) PART_OF
    for m in re.finditer(r"([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})(?:是|属于)([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})的一部分", text):
        add_relation(m.group(1).strip(), m.group(2).strip(), "PART_OF")
    for m in re.finditer(
        r"([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})\s+(?:is part of|part of)\s+([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})",
        text,
        re.IGNORECASE,
    ):
        add_relation(m.group(1).strip(), m.group(2).strip(), "PART_OF")

    # 8) ACQUIRED
    for m in re.finditer(
        r"([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})(?:收购了|并购了|acquired)\s+([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})",
        text,
        re.IGNORECASE,
    ):
        add_relation(m.group(1).strip(), m.group(2).strip(), "ACQUIRED")

    # 9) INVESTED_IN
    for m in re.finditer(
        r"([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})(?:投资了|invested in)\s+([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})",
        text,
        re.IGNORECASE,
    ):
        add_relation(m.group(1).strip(), m.group(2).strip(), "INVESTED_IN")

    # 10) COLLABORATES_WITH
    for m in re.finditer(
        r"([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})(?:与|and)\s*([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})"
        r"(?:合作|协作|collaborated with|collaborates with)",
        text,
        re.IGNORECASE,
    ):
        add_relation(m.group(1).strip(), m.group(2).strip(), "COLLABORATES_WITH")

    # 11) PUBLISHED
    for m in re.finditer(
        r"([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})(?:发表了|published)\s+([\u4e00-\u9fa5A-Za-z&\-\s]{2,40})",
        text,
        re.IGNORECASE,
    ):
        add_relation(m.group(1).strip(), m.group(2).strip(), "PUBLISHED")

    # 12) AWARDED / BORN_IN
    for m in re.finditer(
        r"([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})(?:获得了|won|received)\s+([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})",
        text,
        re.IGNORECASE,
    ):
        add_relation(m.group(1).strip(), m.group(2).strip(), "AWARDED")
    for m in re.finditer(
        r"([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})(?:出生于|was born in)\s+([\u4e00-\u9fa5A-Za-z&\-\s]{2,30})",
        text,
        re.IGNORECASE,
    ):
        add_relation(m.group(1).strip(), m.group(2).strip(), "BORN_IN")

    # 兜底：若文本包含“在...工作”且存在识别到的人和机构，补充最近实体关系
    if re.search(r"在.+工作|works at|works for", text, re.IGNORECASE) and persons and orgs:
        add_relation(persons[0]["text"], orgs[0]["text"], "WORKS_FOR")

    if "会面" in text and len(persons) >= 2:
        add_relation(persons[0]["text"], persons[1]["text"], "MEET_WITH")
        if locs:
            add_relation(persons[0]["text"], locs[0]["text"], "MEET_AT")

    # 指代消解关系：he -> Steve Jobs
    for link in coref_links:
        add_relation(link["mention"], link["antecedent"], "COREFERS_TO")

    return relations


def extract_entities_and_relations(text: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    entities = spacy_extract_entities(text)
    entities, coref_links = resolve_pronoun_coreference(text, entities)
    relations = extract_relations(text, entities, coref_links)
    return entities, relations, coref_links


def build_kg_graph_data(entities: List[Dict], relations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    node_map: Dict[str, Dict] = {}

    def upsert_node(name: str, label: str) -> None:
        if name not in node_map:
            style = GRAPH_STYLE.get(label, GRAPH_STYLE["UNK"])
            node_map[name] = {
                "id": name,
                "name": name,
                "value": ENTITY_LABEL_MAP.get(label, label),
                "symbolSize": style["size"],
                "itemStyle": {"color": style["color"]},
                "draggable": True,
            }

    for ent in entities:
        upsert_node(ent["text"], ent["label"])

    edges: List[Dict] = []
    for rel in relations:
        src = rel["source"]
        tgt = rel["target"]
        relation = rel["relation"]
        if src not in node_map:
            upsert_node(src, "UNK")
        if tgt not in node_map:
            upsert_node(tgt, "UNK")
        edges.append(
            {
                "source": src,
                "target": tgt,
                "value": relation,
                "label": {"show": True, "formatter": relation, "fontSize": 11},
                "lineStyle": {"width": 1.8, "curveness": 0.12, "opacity": 0.9},
            }
        )

    return list(node_map.values()), edges


def render_kg_chart(nodes: List[Dict], edges: List[Dict]) -> None:
    options = {
        "tooltip": {"show": True},
        "animationDurationUpdate": 600,
        "series": [
            {
                "type": "graph",
                "layout": "force",
                "roam": True,
                "draggable": True,
                "focusNodeAdjacency": True,
                "data": nodes,
                "links": edges,
                "edgeSymbol": ["none", "arrow"],
                "edgeSymbolSize": [4, 10],
                "force": {"repulsion": 650, "edgeLength": [120, 240], "gravity": 0.08},
                "label": {"show": True, "position": "right", "fontSize": 12},
                "lineStyle": {"color": "#999"},
            }
        ],
    }
    st_echarts(options=options, height="620px")


def render_highlight_html(text: str, entities: List[Dict]) -> str:
    if not entities:
        return f"<div style='line-height:1.8'>{html.escape(text)}</div>"

    parts = []
    cursor = 0
    for ent in entities:
        start, end = ent["start"], ent["end"]
        if cursor < start:
            parts.append(html.escape(text[cursor:start]))
        color = ENTITY_COLORS.get(ent["label"], "#f1f3f5")
        label_name = ENTITY_LABEL_MAP.get(ent["label"], ent["label"])
        parts.append(
            f"<span style='background:{color};padding:2px 6px;border-radius:6px;margin:0 2px;'>"
            f"{html.escape(text[start:end])}<sup style='font-size:11px;margin-left:4px;'>{label_name}</sup>"
            "</span>"
        )
        cursor = end
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))
    return "<div style='line-height:1.8'>" + "".join(parts) + "</div>"


def to_bio_sequence(text: str, entities: List[Dict]) -> str:
    tags = ["O"] * len(text)
    for ent in entities:
        start, end, label = ent["start"], ent["end"], ent["label"]
        if start < len(text):
            tags[start] = f"B-{label}"
        for i in range(start + 1, min(end, len(text))):
            tags[i] = f"I-{label}"
    return "\n".join(f"{ch}\t{tag}" for ch, tag in zip(text, tags))


st.set_page_config(page_title="NLP 抽取链路演示", layout="wide")
st.title("NLP 抽取链路演示")
st.caption("模块1：实体识别 | 模块2：关系抽取（Subject/Object/Predicate）| 模块3：知识图谱可视化")

page = st.sidebar.radio("选择功能模块", ["模块1：实体识别", "模块2：关系抽取", "模块3：知识图谱"], index=0)

input_text = st.text_area(
    "请输入或粘贴中文/英文语料：",
    value=DEFAULT_TEXT,
    height=180,
    placeholder="在这里输入一段文本...",
)

analyze_clicked = st.button("确认输入并开始分析", type="primary")

if "entities" not in st.session_state:
    st.session_state.entities = []
if "last_text" not in st.session_state:
    st.session_state.last_text = ""
if "relations" not in st.session_state:
    st.session_state.relations = []
if "coref_links" not in st.session_state:
    st.session_state.coref_links = []

nlp_model, nlp_model_name, nlp_load_error = load_spacy_model()
if nlp_model_name:
    st.success(f"当前实体识别引擎：spaCy（{nlp_model_name}）")
else:
    st.warning("未检测到 spaCy 语言模型，当前自动回退到规则抽取。")
    st.info(
        "请在你的虚拟环境中安装模型后重启应用："
        "`python -m spacy download en_core_web_sm` 或 "
        "`python -m spacy download zh_core_web_sm`"
    )
    if nlp_load_error:
        st.caption(f"模型加载信息：{nlp_load_error}")

if analyze_clicked:
    st.session_state.entities, st.session_state.relations, st.session_state.coref_links = extract_entities_and_relations(
        input_text
    )
    st.session_state.last_text = input_text

if st.session_state.last_text:
    entities = st.session_state.entities
    relations = st.session_state.relations
    coref_links = st.session_state.coref_links

    if page == "模块1：实体识别":
        st.info(
            "初学者讲解：BIO 标注中，B（Begin）表示实体起点，I（Inside）表示实体内部连续部分，"
            "O（Outside）表示非实体。模型通过这三类标签学习实体的边界与类型。"
        )
        st.subheader("识别结果")
        compare_view = st.checkbox("同时对比高亮结果与 BIO 标注", value=True, key="compare_view_module1")
        show_bio_only = st.checkbox("仅查看 BIO 标注", value=False, key="show_bio_only_module1")

        if compare_view and not show_bio_only:
            left_col, right_col = st.columns(2)
            with left_col:
                st.markdown("**高亮视图**")
                st.markdown(
                    render_highlight_html(st.session_state.last_text, entities),
                    unsafe_allow_html=True,
                )
            with right_col:
                st.markdown("**BIO 底层标注**")
                st.code(to_bio_sequence(st.session_state.last_text, entities), language="text")
        elif show_bio_only:
            st.code(to_bio_sequence(st.session_state.last_text, entities), language="text")
        else:
            st.markdown(
                render_highlight_html(st.session_state.last_text, entities),
                unsafe_allow_html=True,
            )

        if entities:
            st.subheader("实体列表")
            for i, ent in enumerate(entities, start=1):
                st.write(
                    f"{i}. {ent['text']} | 类别: {ENTITY_LABEL_MAP.get(ent['label'], ent['label'])} "
                    f"| 位置: [{ent['start']}, {ent['end']})"
                )
        else:
            st.info("未识别到实体。你可以尝试输入包含人名、机构名、地名的句子。")
    elif page == "模块2：关系抽取":
        st.info(
            "初学者讲解：关系抽取本质是在图结构里，为两个实体节点预测是否存在一条特定语义边，"
            "例如 WORKS_FOR、FOUNDER_OF。"
        )
        st.subheader("关系抽取结果（SPO）")
        show_relation_debug = st.checkbox("查看关系抽取底层证据", value=False, key="show_debug_module2")
        st.markdown(
            render_highlight_html(st.session_state.last_text, entities),
            unsafe_allow_html=True,
        )
        if relations:
            relation_rows = [
                {
                    "Subject": rel["source"],
                    "Object": rel["target"],
                    "Predicate": rel["relation"],
                }
                for rel in relations
            ]
            st.table(relation_rows)
            if coref_links:
                st.subheader("指代消解结果")
                coref_rows = [
                    {"Mention": item["mention"], "Resolved To": item["antecedent"], "Type": item["label"]}
                    for item in coref_links
                ]
                st.table(coref_rows)
            if show_relation_debug:
                st.subheader("底层证据（调试视图）")
                st.caption("用于解释关系抽取来源：BIO 标注、指代映射、关系原始结构。")
                st.code(to_bio_sequence(st.session_state.last_text, entities), language="text")
                st.write("指代映射原始数据")
                st.json(coref_links)
                st.write("关系三元组原始数据")
                st.json(relations)
        else:
            st.info("当前文本未抽取到关系。可尝试包含“在...工作/创立/会面”等表达的句子。")
            if show_relation_debug:
                st.subheader("底层证据（调试视图）")
                st.code(to_bio_sequence(st.session_state.last_text, entities), language="text")
                st.write("指代映射原始数据")
                st.json(coref_links)
    else:
        st.info(
            "初学者讲解：知识图谱会把线性文本中的实体和关系重组为网状结构数据，"
            "让“谁和谁有什么关系”从句子顺序变成可查询、可推理的图连接。"
        )
        st.subheader("知识图谱可视化")
        st.caption("节点由实体生成，边由关系三元组生成；支持拖拽节点与滚轮缩放。")
        nodes, edges = build_kg_graph_data(entities, relations)
        if not nodes:
            st.info("暂无可视化节点。请先输入文本并完成分析。")
        elif not edges:
            st.warning("已识别实体，但未抽取到关系。当前仅显示离散节点。")
            render_kg_chart(nodes, edges)
        else:
            render_kg_chart(nodes, edges)
            st.write(f"节点数：{len(nodes)} | 边数：{len(edges)}")
