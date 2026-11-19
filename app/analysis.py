"""低频变异区间分析。

针对当前已完成的队列 VCF，按"基因名 + 染色体 + 1 基闭区间"的区间列表
统计样本携带的低频等位基因：

* 仅处理 REF 与所有 ALT 均为单碱基的 SNV 记录，其他记录整条跳过；
* 逐 ALT 用非缺失二倍体 GT 计算 AC/AN（缺失 GT 既不计入 AN 也不计入
  AC，绝不当作 0/0），保留 0 < AC/AN <= 0.25 的等位基因；
* 同一基因的区间重叠时先合并为不相交区间，位点只参与一次统计；不同基因
  的区间相互独立，同一位点可分属多个基因；
* 含至少一个低频等位基因的位点称为候选位点；逐样本给出这些 ALT 的拷贝
  数与有 GT 的候选位点数；候选位点全部缺失 GT 的样本标记为"无法计算"。
"""
from __future__ import annotations

from dataclasses import dataclass

import pysam

from . import config


class AnalysisError(ValueError):
    """Raised when an analysis request is invalid or cannot be started."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class AnalysisAborted(Exception):
    """Raised when the cohort result changes underneath a running analysis."""


@dataclass(frozen=True)
class Interval:
    gene: str
    chrom: str
    start: int
    end: int


# --------------------------------------------------------------------------- #
# Interval parsing and merging
# --------------------------------------------------------------------------- #
def _coerce_coord(value, field: str, gene: str) -> int:
    if isinstance(value, bool):
        raise AnalysisError(f"基因 {gene} 的{field}必须是正整数")
    if isinstance(value, int):
        coord = value
    elif isinstance(value, str) and value.strip().isdigit():
        coord = int(value.strip())
    else:
        raise AnalysisError(f"基因 {gene} 的{field}必须是正整数")
    if not 1 <= coord <= config.MAX_COORD:
        raise AnalysisError(
            f"基因 {gene} 的{field}超出允许范围（1~{config.MAX_COORD}）"
        )
    return coord


def parse_intervals(raw_intervals) -> list[Interval]:
    """Validate the submitted intervals (1-based closed, at most 10)."""
    if not isinstance(raw_intervals, (list, tuple)) or not raw_intervals:
        raise AnalysisError("至少需要填写 1 条区间")
    if len(raw_intervals) > config.MAX_INTERVALS:
        raise AnalysisError(f"区间数量超过上限（最多 {config.MAX_INTERVALS} 条）")
    intervals: list[Interval] = []
    for index, raw in enumerate(raw_intervals, start=1):
        if not isinstance(raw, dict):
            raise AnalysisError(f"第 {index} 条区间格式不正确")
        gene = str(raw.get("gene") or "").strip()
        chrom = str(raw.get("chrom") or "").strip()
        if not gene:
            raise AnalysisError(f"第 {index} 条区间缺少基因名")
        if not chrom:
            raise AnalysisError(f"基因 {gene} 的区间缺少染色体")
        if len(gene) > 64 or len(chrom) > 64:
            raise AnalysisError(f"基因 {gene} 的基因名或染色体过长（限 64 字符）")
        start = _coerce_coord(raw.get("start"), "起点", gene)
        end = _coerce_coord(raw.get("end"), "终点", gene)
        if end < start:
            raise AnalysisError(
                f"基因 {gene} 的区间终点 {end} 小于起点 {start}，要求 1 基闭区间"
            )
        intervals.append(Interval(gene=gene, chrom=chrom, start=start, end=end))
    return intervals


def merge_intervals_by_gene(
    intervals: list[Interval],
) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """Group intervals by gene and merge overlaps per chromosome.

    Returns ``{gene: {chrom: [(start, end), ...]}}`` with sorted, disjoint
    spans so a site inside one gene is counted exactly once no matter how
    many submitted intervals cover it.
    """
    by_gene: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for iv in intervals:
        by_gene.setdefault(iv.gene, {}).setdefault(iv.chrom, []).append(
            (iv.start, iv.end)
        )
    merged: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for gene, chrom_map in by_gene.items():
        merged[gene] = {}
        for chrom, spans in chrom_map.items():
            spans.sort()
            disjoint: list[list[int]] = []
            for start, end in spans:
                if disjoint and start <= disjoint[-1][1] + 1:
                    disjoint[-1][1] = max(disjoint[-1][1], end)
                else:
                    disjoint.append([start, end])
            merged[gene][chrom] = [(s, e) for s, e in disjoint]
    return merged


# --------------------------------------------------------------------------- #
# Cohort scan
# --------------------------------------------------------------------------- #
_ABORT_CHECK_EVERY = 250
_MAX_ALLELE_DETAILS = 500


def _called_gt(sample_data) -> tuple | None:
    """Return the diploid GT tuple, or None when (partly) missing.

    Only fully-called two-allele genotypes participate in AC/AN and copy
    counting; a missing GT is never treated as 0/0.
    """
    gt = sample_data.get("GT")
    if gt is None or len(gt) != 2 or gt[0] is None or gt[1] is None:
        return None
    return gt


def run_analysis(vcf_path: str, intervals: list[Interval], is_aborted=None) -> dict:
    """Scan the cohort VCF and compute per-gene, per-sample low-frequency
    allele carriage.

    ``is_aborted`` is an optional zero-argument callable polled between
    records; returning True raises :class:`AnalysisAborted` so a result
    based on an outdated cohort file is never published.
    """
    merged = merge_intervals_by_gene(intervals)
    genes = list(dict.fromkeys(iv.gene for iv in intervals))  # first appearance

    vcf = pysam.VariantFile(vcf_path)
    try:
        samples = list(vcf.header.samples)
        state = {
            gene: {
                "snv_sites": 0,
                "skipped_non_snv": 0,
                "candidate_sites": 0,
                "alleles": [],
                "alleles_truncated": False,
                "per_sample": {
                    sample: {"alt_copies": 0, "called_sites": 0}
                    for sample in samples
                },
            }
            for gene in genes
        }

        records_seen = 0
        for record in vcf:
            records_seen += 1
            if (
                is_aborted is not None
                and records_seen % _ABORT_CHECK_EVERY == 0
                and is_aborted()
            ):
                raise AnalysisAborted("分析期间队列结果已变更，该在途分析立即失效")

            hit_genes = [
                gene
                for gene in genes
                if record.chrom in merged[gene]
                and any(
                    start <= record.pos <= end
                    for start, end in merged[gene][record.chrom]
                )
            ]
            if not hit_genes:
                continue

            alts = tuple(a for a in (record.alts or ()) if a != ".")
            is_snv = len(record.ref) == 1 and all(len(a) == 1 for a in alts)
            if not is_snv:
                for gene in hit_genes:
                    state[gene]["skipped_non_snv"] += 1
                continue
            for gene in hit_genes:
                state[gene]["snv_sites"] += 1
            if not alts:
                continue

            called_gts = [_called_gt(record.samples[s]) for s in samples]
            an = 2 * sum(1 for gt in called_gts if gt is not None)
            if an == 0:
                continue
            # AC/AN are cohort-wide, so one computation serves every gene
            # whose intervals contain this site.
            qualifying: list[tuple[int, str, int]] = []  # (alt index, alt, ac)
            for alt_index, alt in enumerate(alts, start=1):
                ac = sum(
                    gt.count(alt_index) for gt in called_gts if gt is not None
                )
                if ac > 0 and ac / an <= config.LOW_FREQ_MAX_AF:
                    qualifying.append((alt_index, alt, ac))
            if not qualifying:
                continue

            qualifying_indices = {idx for idx, _, _ in qualifying}
            for gene in hit_genes:
                gene_state = state[gene]
                gene_state["candidate_sites"] += 1
                for alt_index, alt, ac in qualifying:
                    if len(gene_state["alleles"]) < _MAX_ALLELE_DETAILS:
                        gene_state["alleles"].append(
                            {
                                "chrom": record.chrom,
                                "pos": record.pos,
                                "ref": record.ref,
                                "alt": alt,
                                "ac": ac,
                                "an": an,
                                "af": ac / an,
                            }
                        )
                    else:
                        gene_state["alleles_truncated"] = True
                per_sample = gene_state["per_sample"]
                for sample, gt in zip(samples, called_gts):
                    if gt is None:
                        # 缺失 GT：该候选位点不可判，绝不当作 0/0。
                        continue
                    entry = per_sample[sample]
                    entry["called_sites"] += 1
                    entry["alt_copies"] += sum(
                        1 for allele in gt if allele in qualifying_indices
                    )
    finally:
        vcf.close()

    if is_aborted is not None and is_aborted():
        raise AnalysisAborted("分析期间队列结果已变更，该在途分析立即失效")

    return {
        "samples": samples,
        "genes": [
            {
                "gene": gene,
                "intervals": [
                    {"chrom": chrom, "start": start, "end": end}
                    for chrom, spans in merged[gene].items()
                    for start, end in spans
                ],
                "snv_sites": state[gene]["snv_sites"],
                "skipped_non_snv": state[gene]["skipped_non_snv"],
                "candidate_sites": state[gene]["candidate_sites"],
                "alleles": state[gene]["alleles"],
                "alleles_truncated": state[gene]["alleles_truncated"],
                "per_sample": {
                    sample: {
                        "alt_copies": counts["alt_copies"],
                        "called_sites": counts["called_sites"],
                        "computable": counts["called_sites"] > 0,
                    }
                    for sample, counts in state[gene]["per_sample"].items()
                },
            }
            for gene in genes
        ],
    }
