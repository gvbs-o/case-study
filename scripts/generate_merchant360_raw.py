"""
Gera a camada raw ficticia do projeto Merchant 360.

Saidas:
  data/raw/estabelecimentos/estabelecimentos.parquet
  data/raw/faturamento/faturamento.parquet
  data/raw/transacoes/transacoes.parquet
  data/raw/alertas_antifraude/alertas_antifraude.parquet
  data/raw/fraudes/fraudes.parquet
  data/raw/solicitacoes_central/solicitacoes_central.parquet
  data/raw/ciclo_vida/ciclo_vida.parquet
  data/raw/interacoes_relacionamento/interacoes_relacionamento.parquet

Execute na raiz do projeto:
    python scripts/generate_merchant360_raw.py
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"

SEED = 360
N_MERCHANTS = 380
MONTHS = pd.period_range("2024-06", "2026-05", freq="M")

random.seed(SEED)
np.random.seed(SEED)


@dataclass(frozen=True)
class ProfileRule:
    weight: float
    status_weights: dict[str, float]
    risco_inicial: str
    risco_atual: str
    volume_multiplier: float
    growth_monthly: float
    volatility: float
    alert_rate: float
    fraud_rate: float
    request_rate: float


PROFILE_RULES = {
    "saudavel": ProfileRule(
        0.32,
        {"Ativo": 0.95, "Retido": 0.05},
        "Baixo",
        "Baixo",
        1.10,
        0.012,
        0.08,
        0.025,
        0.0035,
        0.055,
    ),
    "em crescimento": ProfileRule(
        0.22,
        {"Ativo": 0.93, "Retido": 0.07},
        "Baixo",
        "Medio",
        0.85,
        0.055,
        0.14,
        0.055,
        0.0070,
        0.080,
    ),
    "instavel": ProfileRule(
        0.18,
        {"Ativo": 0.72, "Retido": 0.20, "Bloqueado": 0.08},
        "Medio",
        "Medio",
        0.95,
        -0.002,
        0.35,
        0.090,
        0.0120,
        0.130,
    ),
    "alto risco": ProfileRule(
        0.14,
        {"Ativo": 0.58, "Retido": 0.22, "Bloqueado": 0.20},
        "Medio",
        "Alto",
        0.90,
        0.020,
        0.28,
        0.180,
        0.0300,
        0.190,
    ),
    "bloqueado": ProfileRule(
        0.08,
        {"Bloqueado": 0.86, "Retido": 0.14},
        "Alto",
        "Alto",
        0.70,
        -0.018,
        0.30,
        0.220,
        0.0420,
        0.230,
    ),
    "encerrado": ProfileRule(
        0.06,
        {"Encerrado": 1.00},
        "Medio",
        "Alto",
        0.55,
        -0.030,
        0.22,
        0.120,
        0.0180,
        0.160,
    ),
}

SEGMENTOS = [
    "Alimentacao",
    "Moda",
    "Farmacia",
    "Pet Shop",
    "Servicos",
    "Educacao",
    "Saude",
    "Autopecas",
    "Mercado",
    "Tecnologia",
    "Turismo",
    "Construção",
]
UF_CIDADES = {
    "SP": ["Sao Paulo", "Campinas", "Santos", "Ribeirao Preto"],
    "RJ": ["Rio de Janeiro", "Niteroi", "Petropolis"],
    "MG": ["Belo Horizonte", "Uberlandia", "Juiz de Fora"],
    "PR": ["Curitiba", "Londrina", "Maringa"],
    "RS": ["Porto Alegre", "Caxias do Sul", "Pelotas"],
    "SC": ["Florianopolis", "Joinville", "Blumenau"],
    "BA": ["Salvador", "Feira de Santana"],
    "PE": ["Recife", "Olinda"],
    "GO": ["Goiania", "Anapolis"],
    "CE": ["Fortaleza", "Juazeiro do Norte"],
}
CANAIS = ["POS", "E-commerce", "Link de pagamento", "TEF", "App"]
PRODUTOS = ["Credito", "Debito", "Pix", "Voucher", "Antecipacao"]
BANDEIRAS = ["Visa", "Mastercard", "Elo", "Amex", "Hipercard"]
EXECUTIVOS = [
    "Ana Souza",
    "Bruno Lima",
    "Carla Rocha",
    "Daniel Pires",
    "Eduarda Mello",
    "Felipe Tavares",
    "Gabriela Nunes",
    "Henrique Alves",
]


def ensure_lake_dirs() -> None:
    for layer in ["raw", "bronze", "silver", "gold"]:
        (DATA / layer).mkdir(parents=True, exist_ok=True)
    for domain in [
        "estabelecimentos",
        "faturamento",
        "transacoes",
        "alertas_antifraude",
        "fraudes",
        "solicitacoes_central",
        "ciclo_vida",
        "interacoes_relacionamento",
    ]:
        (RAW / domain).mkdir(parents=True, exist_ok=True)


def weighted_choice(weights: dict[str, float]) -> str:
    labels = list(weights)
    probs = list(weights.values())
    return random.choices(labels, weights=probs, k=1)[0]


def fake_cnpj(i: int) -> str:
    base = f"{10_000_000 + i:08d}{random.randint(1, 9999):04d}{random.randint(0, 99):02d}"
    return f"{base[:2]}.{base[2:5]}.{base[5:8]}/{base[8:12]}-{base[12:]}"


def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(days=random.randint(0, max(delta.days, 1)))


def build_estabelecimentos() -> pd.DataFrame:
    profile_names = list(PROFILE_RULES)
    profile_weights = [PROFILE_RULES[p].weight for p in profile_names]
    rows = []

    for i in range(1, N_MERCHANTS + 1):
        profile = random.choices(profile_names, weights=profile_weights, k=1)[0]
        rule = PROFILE_RULES[profile]
        uf = random.choice(list(UF_CIDADES))
        status = weighted_choice(rule.status_weights)
        segmento = random.choice(SEGMENTOS)
        porte = random.choices(["MEI", "Pequeno", "Medio", "Grande"], weights=[2, 5, 3, 1], k=1)[0]
        data_cred = random_date(datetime(2018, 1, 1), datetime(2024, 5, 20)).date()

        rows.append(
            {
                "merchant_id": f"MRC{i:06d}",
                "cnpj": fake_cnpj(i),
                "razao_social": f"{segmento} {i:04d} Comercio Ltda",
                "nome_fantasia": f"{segmento} {random.choice(['Prime', 'Mais', 'Center', 'Brasil', 'Digital'])} {i:04d}",
                "segmento": segmento,
                "canal_credenciamento": random.choice(["Agencia", "Parceiro", "Online", "Executivo Comercial"]),
                "cidade": random.choice(UF_CIDADES[uf]),
                "uf": uf,
                "data_credenciamento": data_cred,
                "status_atual": status,
                "porte": porte,
                "risco_inicial": rule.risco_inicial,
                "risco_atual": rule.risco_atual,
                "executivo_responsavel": random.choice(EXECUTIVOS),
                "produto_principal": random.choice(PRODUTOS),
                "perfil_geracao": profile,
            }
        )
    return pd.DataFrame(rows)


def merchant_month_shape(estabelecimentos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    risky_growth_ids = set(
        estabelecimentos[estabelecimentos["perfil_geracao"].isin(["em crescimento", "alto risco"])]
        .sample(28, random_state=SEED)["merchant_id"]
        .tolist()
    )
    intervention_month = {m: random.randint(13, 19) for m in risky_growth_ids}
    status_month = {
        row.merchant_id: random.randint(12, 20)
        for row in estabelecimentos.itertuples()
        if row.status_atual in {"Bloqueado", "Encerrado", "Retido"}
    }

    for row in estabelecimentos.itertuples():
        rule = PROFILE_RULES[row.perfil_geracao]
        base_qtd = int(
            random.choice([12, 16, 20, 24, 30, 40, 55])
            * rule.volume_multiplier
            * {"MEI": 0.55, "Pequeno": 0.85, "Medio": 1.20, "Grande": 2.20}[row.porte]
        )
        base_ticket = random.uniform(45, 420) * {"MEI": 0.80, "Pequeno": 1.0, "Medio": 1.30, "Grande": 1.75}[row.porte]

        for idx, month in enumerate(MONTHS):
            seasonal = 1 + 0.18 * math.sin((idx % 12) / 12 * 2 * math.pi)
            trend = (1 + rule.growth_monthly) ** idx
            if row.merchant_id in risky_growth_ids and idx >= intervention_month[row.merchant_id] - 5:
                trend *= 1 + min((idx - intervention_month[row.merchant_id] + 6) * 0.18, 1.25)
            if row.merchant_id in status_month and idx >= status_month[row.merchant_id]:
                if row.status_atual == "Bloqueado":
                    trend *= 0.32
                elif row.status_atual == "Encerrado":
                    trend *= 0.06 if idx > status_month[row.merchant_id] else 0.20
                elif row.status_atual == "Retido":
                    trend *= 0.58

            noise = max(0.12, np.random.normal(1, rule.volatility))
            qtd = max(1, int(base_qtd * seasonal * trend * noise))
            if row.status_atual == "Encerrado" and idx > status_month.get(row.merchant_id, 18) + 2:
                qtd = random.choice([0, 0, 1])
            rows.append(
                {
                    "merchant_id": row.merchant_id,
                    "competencia": str(month),
                    "qtd_prevista": qtd,
                    "ticket_base": base_ticket * np.random.normal(1, 0.12),
                    "perfil_geracao": row.perfil_geracao,
                    "risky_growth": row.merchant_id in risky_growth_ids,
                    "intervention_idx": intervention_month.get(row.merchant_id),
                    "status_idx": status_month.get(row.merchant_id),
                    "status_geracao": row.status_atual,
                }
            )
    return pd.DataFrame(rows)


def build_transacoes(month_shape: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seq = 1
    for mm in month_shape.itertuples():
        period = pd.Period(mm.competencia, freq="M")
        start = period.start_time.to_pydatetime()
        days_in_month = period.days_in_month
        rule = PROFILE_RULES[mm.perfil_geracao]

        for _ in range(int(mm.qtd_prevista)):
            dt = start + timedelta(
                days=random.randint(0, days_in_month - 1),
                hours=random.randint(7, 22),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
            )
            valor = max(3.5, np.random.gamma(shape=2.2, scale=max(mm.ticket_base / 2.2, 2)))
            score = np.clip(
                np.random.normal(
                    18 + rule.alert_rate * 220 + rule.fraud_rate * 600,
                    16 + rule.volatility * 32,
                ),
                0,
                100,
            )
            if mm.risky_growth and mm.intervention_idx and MONTHS.get_loc(period) >= mm.intervention_idx - 4:
                score = min(100, score + random.uniform(10, 28))

            status = random.choices(
                ["Aprovada", "Negada", "Cancelada", "Chargeback"],
                weights=[93 - min(score / 5, 14), 4 + score / 18, 2, 1 + score / 30],
                k=1,
            )[0]
            rows.append(
                {
                    "transaction_id": f"TRX{seq:010d}",
                    "merchant_id": mm.merchant_id,
                    "data_transacao": dt,
                    "valor_transacao": round(float(valor), 2),
                    "bandeira": random.choice(BANDEIRAS),
                    "produto": random.choices(PRODUTOS, weights=[55, 24, 12, 6, 3], k=1)[0],
                    "canal": random.choices(CANAIS, weights=[45, 25, 14, 12, 4], k=1)[0],
                    "pais_emissor": random.choices(["BR", "US", "AR", "CL", "PT", "MX"], weights=[92, 2, 2, 1, 1, 2], k=1)[0],
                    "status_transacao": status,
                    "forma_captura": random.choices(["Chip", "NFC", "Digitada", "QR Code", "Tokenizada"], weights=[40, 28, 9, 13, 10], k=1)[0],
                    "qtd_parcelas": int(random.choices([1, 2, 3, 4, 6, 10, 12], weights=[58, 9, 11, 6, 7, 5, 4], k=1)[0]),
                    "score_risco_transacao": round(float(score), 2),
                }
            )
            seq += 1
    return pd.DataFrame(rows)


def build_faturamento(transacoes: pd.DataFrame) -> pd.DataFrame:
    tx = transacoes[transacoes["status_transacao"].isin(["Aprovada", "Chargeback"])].copy()
    tx["competencia"] = tx["data_transacao"].dt.to_period("M").astype(str)
    grouped = (
        tx.groupby(["merchant_id", "competencia"])
        .agg(tpv_total=("valor_transacao", "sum"), qtd_transacoes=("transaction_id", "count"))
        .reset_index()
    )
    all_months = pd.MultiIndex.from_product(
        [transacoes["merchant_id"].unique(), [str(m) for m in MONTHS]],
        names=["merchant_id", "competencia"],
    ).to_frame(index=False)
    df = all_months.merge(grouped, on=["merchant_id", "competencia"], how="left").fillna({"tpv_total": 0, "qtd_transacoes": 0})
    df["qtd_transacoes"] = df["qtd_transacoes"].astype(int)
    df["ticket_medio"] = np.where(df["qtd_transacoes"] > 0, df["tpv_total"] / df["qtd_transacoes"], 0).round(2)
    df["tpv_total"] = (df["ticket_medio"] * df["qtd_transacoes"]).round(2)
    df["mdr_medio"] = np.random.uniform(0.012, 0.035, len(df)).round(4)
    df["receita_liquida"] = (df["tpv_total"] * df["mdr_medio"] * np.random.uniform(0.84, 0.96, len(df))).round(2)
    df["canal_principal"] = np.random.choice(CANAIS, len(df), p=[0.45, 0.25, 0.14, 0.12, 0.04])

    df = df.sort_values(["merchant_id", "competencia"])
    prev = df.groupby("merchant_id")["tpv_total"].shift(1)
    prev_y = df.groupby("merchant_id")["tpv_total"].shift(12)
    df["crescimento_tpv_mom"] = np.where(prev > 0, (df["tpv_total"] - prev) / prev, np.nan)
    df["crescimento_tpv_yoy"] = np.where(prev_y > 0, (df["tpv_total"] - prev_y) / prev_y, np.nan)
    df[["crescimento_tpv_mom", "crescimento_tpv_yoy"]] = df[["crescimento_tpv_mom", "crescimento_tpv_yoy"]].round(4)
    return df[
        [
            "merchant_id",
            "competencia",
            "tpv_total",
            "qtd_transacoes",
            "ticket_medio",
            "receita_liquida",
            "mdr_medio",
            "crescimento_tpv_mom",
            "crescimento_tpv_yoy",
            "canal_principal",
        ]
    ]


def build_alertas(transacoes: pd.DataFrame, estabelecimentos: pd.DataFrame) -> pd.DataFrame:
    profiles = estabelecimentos[["merchant_id", "perfil_geracao"]]
    tx = transacoes.merge(profiles, on="merchant_id", how="left")
    base = tx["score_risco_transacao"] / 100
    profile_boost = tx["perfil_geracao"].map({p: r.alert_rate for p, r in PROFILE_RULES.items()})
    tx["_peso"] = (base**2 + profile_boost).clip(0.001, None)
    n_alerts = min(14500, len(tx))
    sample = tx.sample(n=n_alerts, weights="_peso", random_state=SEED + 1, replace=False)

    rows = []
    for i, row in enumerate(sample.itertuples(), start=1):
        score = min(100, max(1, row.score_risco_transacao + np.random.normal(8, 11)))
        data_alerta = row.data_transacao + timedelta(minutes=random.randint(2, 360), hours=random.randint(0, 72))
        severidade = "Alta" if score >= 78 else "Media" if score >= 48 else "Baixa"
        decisao = random.choices(
            ["Liberar", "Monitorar", "Reter pagamento", "Bloquear transacao", "Escalar caso"],
            weights=[20, 38, 18, 14, 10] if severidade != "Alta" else [6, 22, 27, 25, 20],
            k=1,
        )[0]
        rows.append(
            {
                "alert_id": f"ALR{i:08d}",
                "merchant_id": row.merchant_id,
                "transaction_id": row.transaction_id,
                "data_alerta": data_alerta,
                "tipo_alerta": random.choice(["Velocity", "Chargeback", "Pais emissor incomum", "Ticket atipico", "MCC sensivel"]),
                "regra_disparada": random.choice(["R001-score-alto", "R014-velocity-dia", "R022-ticket-fora-padrao", "R031-emissor-risco"]),
                "severidade": severidade,
                "score_alerta": round(float(score), 2),
                "status_alerta": random.choices(["Aberto", "Em analise", "Concluido", "Falso positivo"], weights=[12, 18, 58, 12], k=1)[0],
                "analista_responsavel": random.choice(["Marina Reis", "Nicolas Prado", "Olivia Castro", "Paulo Mendes", "Rafael Borges"]),
                "decisao": decisao,
                "motivo_decisao": random.choice(["Padrao transacional atipico", "Historico do merchant", "Contestacao recente", "Divergencia operacional"]),
            }
        )
    return pd.DataFrame(rows)


def build_fraudes(transacoes: pd.DataFrame, alertas: pd.DataFrame, estabelecimentos: pd.DataFrame) -> pd.DataFrame:
    profiles = estabelecimentos[["merchant_id", "perfil_geracao"]]
    tx = transacoes.merge(profiles, on="merchant_id", how="left")
    tx = tx[tx["status_transacao"].isin(["Aprovada", "Chargeback"])].copy()
    profile_boost = tx["perfil_geracao"].map({p: r.fraud_rate for p, r in PROFILE_RULES.items()})
    tx["_peso"] = ((tx["score_risco_transacao"] / 100) ** 2.3 + profile_boost).clip(0.0001, None)

    n_frauds = 1850
    alerted_ids = set(alertas.sample(frac=0.42, random_state=SEED + 2)["transaction_id"])
    alerted_tx = tx[tx["transaction_id"].isin(alerted_ids)]
    non_alerted_tx = tx[~tx["transaction_id"].isin(alerted_ids)]
    fraud_tx = pd.concat(
        [
            alerted_tx.sample(n=min(980, len(alerted_tx)), weights="_peso", random_state=SEED + 3, replace=False),
            non_alerted_tx.sample(n=n_frauds - min(980, len(alerted_tx)), weights="_peso", random_state=SEED + 4, replace=False),
        ],
        ignore_index=True,
    )

    rows = []
    for i, row in enumerate(fraud_tx.itertuples(), start=1):
        data_fraude = row.data_transacao + timedelta(days=random.randint(0, 18))
        dias_conf = random.randint(1, 45)
        valor_recuperado = round(float(row.valor_transacao) * random.choices([0, 0.25, 0.5, 0.8, 1], weights=[38, 18, 18, 16, 10], k=1)[0], 2)
        valor_fraude = round(float(row.valor_transacao), 2)
        rows.append(
            {
                "fraud_id": f"FRD{i:08d}",
                "merchant_id": row.merchant_id,
                "transaction_id": row.transaction_id,
                "data_fraude": data_fraude,
                "data_confirmacao": data_fraude + timedelta(days=dias_conf),
                "valor_fraude": valor_fraude,
                "tipo_fraude": random.choice(["Chargeback fraudulento", "Cartao clonado", "Autofraude", "Conta laranja", "Compra nao reconhecida"]),
                "origem_identificacao": "Alerta antifraude" if row.transaction_id in alerted_ids else random.choice(["Contestacao portador", "Bandeira", "Auditoria", "Central"]),
                "status_recuperacao": "Recuperado" if valor_recuperado == valor_fraude else "Parcial" if valor_recuperado > 0 else "Nao recuperado",
                "valor_recuperado": valor_recuperado,
                "perda_liquida": round(valor_fraude - valor_recuperado, 2),
                "dias_ate_confirmacao": dias_conf,
            }
        )
    return pd.DataFrame(rows)


def build_solicitacoes(estabelecimentos: pd.DataFrame, fraudes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seq = 1
    fraud_merchants = set(fraudes["merchant_id"])
    for row in estabelecimentos.itertuples():
        rule = PROFILE_RULES[row.perfil_geracao]
        qtd = max(4, np.random.poisson(22 * (1 + rule.request_rate * 4)))
        if row.merchant_id in fraud_merchants:
            qtd += random.randint(5, 18)
        for _ in range(int(qtd)):
            month = random.choice(MONTHS)
            abertura = month.start_time.to_pydatetime() + timedelta(days=random.randint(0, month.days_in_month - 1), hours=random.randint(8, 20))
            categoria = random.choices(
                ["Contestacao", "Financeiro", "Tecnico", "Cadastro", "Risco", "Comercial"],
                weights=[8 + rule.fraud_rate * 380, 24, 26, 14, 8 + rule.alert_rate * 120, 20],
                k=1,
            )[0]
            subcats = {
                "Contestacao": ["Chargeback", "Venda nao reconhecida", "Comprovante"],
                "Financeiro": ["Liquidacao", "Antecipacao", "Taxas", "Extrato"],
                "Tecnico": ["POS inoperante", "Integracao", "Instabilidade"],
                "Cadastro": ["Alteracao bancaria", "Dados cadastrais", "Credenciamento"],
                "Risco": ["Bloqueio", "Retencao", "Analise antifraude"],
                "Comercial": ["Negociacao MDR", "Upgrade plano", "Produto adicional"],
            }
            prioridade = random.choices(["Baixa", "Media", "Alta", "Critica"], weights=[34, 42, 18, 6], k=1)[0]
            sla = {"Baixa": 72, "Media": 48, "Alta": 24, "Critica": 8}[prioridade]
            resolucao = max(1, int(np.random.gamma(2.1, sla / 3.2)))
            if categoria in {"Contestacao", "Risco"} and row.perfil_geracao in {"alto risco", "bloqueado"}:
                resolucao = int(resolucao * random.uniform(1.2, 2.4))
            status = "Aberto" if abertura > datetime(2026, 5, 15) and random.random() < 0.35 else random.choice(["Resolvido", "Fechado", "Cancelado"])
            fechamento = pd.NaT if status == "Aberto" else abertura + timedelta(hours=resolucao)
            rows.append(
                {
                    "request_id": f"REQ{seq:09d}",
                    "merchant_id": row.merchant_id,
                    "data_abertura": abertura,
                    "data_fechamento": fechamento,
                    "categoria": categoria,
                    "subcategoria": random.choice(subcats[categoria]),
                    "canal_atendimento": random.choice(["Portal", "Telefone", "E-mail", "WhatsApp", "Executivo"]),
                    "status": status,
                    "prioridade": prioridade,
                    "sla_horas": sla,
                    "tempo_resolucao_horas": None if status == "Aberto" else resolucao,
                    "dentro_sla": None if status == "Aberto" else bool(resolucao <= sla),
                    "sentimento_estimado": random.choices(["Positivo", "Neutro", "Negativo"], weights=[18, 55, 27] if categoria in {"Contestacao", "Risco"} else [32, 54, 14], k=1)[0],
                    "descricao_resumida": f"Solicitacao de {categoria.lower()} sobre {random.choice(subcats[categoria]).lower()}.",
                }
            )
            seq += 1
    return pd.DataFrame(rows)


def build_ciclo_vida(estabelecimentos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seq = 1
    for row in estabelecimentos.itertuples():
        cred = pd.Timestamp(row.data_credenciamento).to_pydatetime()
        events = [
            (cred, "Credenciamento", "Cadastro aprovado", "Concluido", 0, "Entrada no portfolio"),
            (cred + timedelta(days=random.randint(7, 45)), "Ativacao", "Primeira transacao", "Concluido", random.uniform(500, 2500), "Merchant iniciou processamento"),
        ]
        if row.perfil_geracao == "em crescimento":
            events.append((datetime(2025, random.randint(2, 11), random.randint(1, 24)), "Expansao", "Aumento relevante de TPV", "Concluido", random.uniform(8000, 65000), "Crescimento acima da media"))
        if row.risco_atual == "Alto":
            events.append((datetime(2025, random.randint(5, 12), random.randint(1, 24)), "Risco", "Revisao antifraude", "Concluido", random.uniform(-12000, -1000), "Risco elevado identificado"))
        if row.status_atual == "Retido":
            events.append((datetime(2026, random.randint(1, 5), random.randint(1, 20)), "Retencao", "Retencao preventiva", "Concluido", random.uniform(-9000, -500), "Liquidacao retida para analise"))
        if row.status_atual == "Bloqueado":
            events.append((datetime(2026, random.randint(1, 5), random.randint(1, 20)), "Bloqueio", "Bloqueio operacional", "Concluido", random.uniform(-50000, -6000), "Bloqueio por suspeita de fraude"))
        if row.status_atual == "Encerrado":
            events.append((datetime(2026, random.randint(1, 5), random.randint(1, 20)), "Encerramento", "Contrato encerrado", "Concluido", random.uniform(-70000, -8000), "Encerramento de relacionamento"))

        for dt, etapa, evento, status, impacto, obs in sorted(events, key=lambda x: x[0]):
            rows.append(
                {
                    "lifecycle_event_id": f"LCF{seq:08d}",
                    "merchant_id": row.merchant_id,
                    "data_evento": dt,
                    "etapa": etapa,
                    "evento": evento,
                    "status_evento": status,
                    "valor_impacto_estimado": round(float(impacto), 2),
                    "observacao": obs,
                }
            )
            seq += 1
    return pd.DataFrame(rows)


def build_interacoes(estabelecimentos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seq = 1
    for row in estabelecimentos.itertuples():
        rule = PROFILE_RULES[row.perfil_geracao]
        qtd = max(3, np.random.poisson(9 + rule.alert_rate * 28))
        for _ in range(int(qtd)):
            month = random.choice(MONTHS)
            data = month.start_time.to_pydatetime() + timedelta(days=random.randint(0, month.days_in_month - 1), hours=random.randint(8, 19))
            motivo = random.choices(
                ["Acompanhamento comercial", "Revisao de taxas", "Analise de risco", "Recuperacao de volume", "Suporte operacional", "Retencao"],
                weights=[28, 18, 8 + rule.alert_rate * 80, 18, 20, 8],
                k=1,
            )[0]
            rows.append(
                {
                    "interaction_id": f"INT{seq:09d}",
                    "merchant_id": row.merchant_id,
                    "data_interacao": data,
                    "area_responsavel": random.choice(["Comercial", "Relacionamento", "Risco", "Operacoes", "Financeiro"]),
                    "tipo_interacao": random.choice(["Ligacao", "E-mail", "Reuniao", "WhatsApp", "Visita"]),
                    "motivo": motivo,
                    "resultado": random.choice(["Sem pendencia", "Plano de acao", "Aguardando retorno", "Escalado", "Resolvido"]),
                    "proxima_acao": random.choice(["Monitorar", "Enviar proposta", "Reavaliar risco", "Agendar retorno", "Sem acao"]),
                    "responsavel": random.choice(EXECUTIVOS),
                }
            )
            seq += 1
    return pd.DataFrame(rows)


def write_raw(df: pd.DataFrame, domain: str, filename: str) -> None:
    df.to_parquet(RAW / domain / filename, index=False)


def validate_with_duckdb(paths: dict[str, Path]) -> dict[str, bool]:
    con = duckdb.connect(database=":memory:")
    for table, path in paths.items():
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path.as_posix()}')")

    merchant_bad = con.execute(
        """
        SELECT count(*) FROM (
            SELECT merchant_id FROM faturamento
            UNION ALL SELECT merchant_id FROM transacoes
            UNION ALL SELECT merchant_id FROM alertas_antifraude
            UNION ALL SELECT merchant_id FROM fraudes
            UNION ALL SELECT merchant_id FROM solicitacoes_central
            UNION ALL SELECT merchant_id FROM ciclo_vida
            UNION ALL SELECT merchant_id FROM interacoes_relacionamento
        ) f
        LEFT JOIN estabelecimentos e USING (merchant_id)
        WHERE e.merchant_id IS NULL
        """
    ).fetchone()[0]
    tx_bad = con.execute(
        """
        SELECT count(*) FROM (
            SELECT transaction_id FROM alertas_antifraude WHERE transaction_id IS NOT NULL
            UNION ALL
            SELECT transaction_id FROM fraudes WHERE transaction_id IS NOT NULL
        ) f
        LEFT JOIN transacoes t USING (transaction_id)
        WHERE t.transaction_id IS NULL
        """
    ).fetchone()[0]
    date_bad = con.execute(
        """
        SELECT count(*)
        FROM fraudes f
        JOIN transacoes t USING (transaction_id)
        WHERE f.data_confirmacao < t.data_transacao
           OR f.data_fraude < t.data_transacao
        """
    ).fetchone()[0]
    finance_bad = con.execute(
        """
        SELECT count(*)
        FROM faturamento
        WHERE abs(ticket_medio - round(tpv_total / nullif(qtd_transacoes, 0), 2)) > 0.01
          AND qtd_transacoes > 0
        """
    ).fetchone()[0]
    loss_bad = con.execute(
        "SELECT count(*) FROM fraudes WHERE abs(perda_liquida - (valor_fraude - valor_recuperado)) > 0.01"
    ).fetchone()[0]
    lifecycle_bad = con.execute(
        """
        SELECT count(*)
        FROM estabelecimentos e
        LEFT JOIN ciclo_vida c
          ON e.merchant_id = c.merchant_id
         AND (
              (e.status_atual = 'Bloqueado' AND c.etapa = 'Bloqueio')
           OR (e.status_atual = 'Encerrado' AND c.etapa = 'Encerramento')
           OR (e.status_atual = 'Retido' AND c.etapa = 'Retencao')
         )
        WHERE e.status_atual IN ('Bloqueado', 'Encerrado', 'Retido')
          AND c.merchant_id IS NULL
        """
    ).fetchone()[0]
    con.close()

    return {
        "merchant_id consistente": merchant_bad == 0,
        "transaction_id consistente": tx_bad == 0,
        "datas coerentes": date_bad == 0,
        "metricas financeiras coerentes": finance_bad == 0 and loss_bad == 0,
        "ciclo de vida consistente": lifecycle_bad == 0,
    }


def main() -> None:
    ensure_lake_dirs()

    estabelecimentos = build_estabelecimentos()
    month_shape = merchant_month_shape(estabelecimentos)
    transacoes = build_transacoes(month_shape)
    faturamento = build_faturamento(transacoes)
    alertas = build_alertas(transacoes, estabelecimentos)
    fraudes = build_fraudes(transacoes, alertas, estabelecimentos)
    solicitacoes = build_solicitacoes(estabelecimentos, fraudes)
    ciclo_vida = build_ciclo_vida(estabelecimentos)
    interacoes = build_interacoes(estabelecimentos)

    estabelecimentos_out = estabelecimentos.drop(columns=["perfil_geracao"])
    outputs = {
        "estabelecimentos": (estabelecimentos_out, "estabelecimentos", "estabelecimentos.parquet"),
        "faturamento": (faturamento, "faturamento", "faturamento.parquet"),
        "transacoes": (transacoes, "transacoes", "transacoes.parquet"),
        "alertas_antifraude": (alertas, "alertas_antifraude", "alertas_antifraude.parquet"),
        "fraudes": (fraudes, "fraudes", "fraudes.parquet"),
        "solicitacoes_central": (solicitacoes, "solicitacoes_central", "solicitacoes_central.parquet"),
        "ciclo_vida": (ciclo_vida, "ciclo_vida", "ciclo_vida.parquet"),
        "interacoes_relacionamento": (interacoes, "interacoes_relacionamento", "interacoes_relacionamento.parquet"),
    }
    paths = {}
    for name, (df, domain, filename) in outputs.items():
        write_raw(df, domain, filename)
        paths[name] = RAW / domain / filename

    validations = validate_with_duckdb(paths)
    if not all(validations.values()):
        failed = ", ".join(k for k, ok in validations.items() if not ok)
        raise RuntimeError(f"Validacoes falharam: {failed}")

    print("Camada raw gerada com sucesso.")
    print()
    print("Tabelas criadas:")
    for name, (df, _, _) in outputs.items():
        print(f"- {name}: {len(df)} registros")
    print()
    print("Periodo coberto:")
    print(f"{MONTHS[0]} ate {MONTHS[-1]}")
    print()
    print("Merchants gerados:")
    print(len(estabelecimentos_out))
    print()
    print("Validacoes:")
    print("- merchant_id consistente: OK")
    print("- transaction_id consistente: OK")
    print("- datas coerentes: OK")
    print("- metricas financeiras coerentes: OK")


if __name__ == "__main__":
    main()
