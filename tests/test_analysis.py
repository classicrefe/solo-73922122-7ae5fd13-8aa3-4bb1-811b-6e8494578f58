"""低频变异区间分析：解析、合并、AC/AN 统计与 API 工作流测试。"""
from __future__ import annotations

import textwrap
import time

import pytest

from app.analysis import (
    AnalysisAborted,
    AnalysisError,
    Interval,
    merge_intervals_by_gene,
    parse_intervals,
    run_analysis,
)
from tests.conftest import DEMO_DIR
from tests.test_api import upload_files, wait_for_terminal

HEADER = textwrap.dedent(
    """\
    ##fileformat=VCFv4.2
    ##reference=GRCh38.demo
    ##contig=<ID=chr1,length=100000>
    ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
    """
)


def vcf_with(path, samples: list[str], rows: list[str]) -> None:
    header = (
        HEADER
        + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(samples)
        + "\n"
    )
    path.write_text(header + "\n".join(rows) + "\n")


@pytest.fixture()
def unit_vcf(tmp_path):
    """4 样本的判定用 VCF：低频/高频/全缺失/非 SNV/无 ALT 全覆盖。"""
    path = tmp_path / "cohort.vcf"
    vcf_with(
        path,
        ["W", "X", "Y", "Z"],
        [
            # 候选：AC=1, AN=6（X 缺失不计入）-> 1/6 <= 0.25
            "chr1\t100\t.\tA\tG\t.\t.\t.\tGT\t0/1\t./.\t0/0\t0/0",
            # 候选：AC=1, AN=4 -> 0.25（边界，含）
            "chr1\t150\t.\tA\tT\t.\t.\t.\tGT\t./.\t./.\t0/1\t0/0",
            # AC=6, AN=6 -> 1.0 排除
            "chr1\t200\t.\tC\tA\t.\t.\t.\tGT\t1/1\t1/1\t1/1\t./.",
            # AC=2, AN=4 -> 0.5 排除；若把缺失当 0/0 则 AN=8、AC/AN=0.25 会
            # 被错误纳入，此行为关键断言点
            "chr1\t250\t.\tC\tT\t.\t.\t.\tGT\t0/1\t0/1\t./.\t./.",
            # AC=7, AN=8 -> 0.875 排除
            "chr1\t300\t.\tG\tA\t.\t.\t.\tGT\t1/1\t1/1\t0/1\t1/1",
            # REF 长度 2 -> 非 SNV，整条跳过
            "chr1\t400\t.\tTT\tA\t.\t.\t.\tGT\t0/1\t0/0\t0/0\t0/0",
            # 一个 ALT 长度 2 -> 非 SNV，整条跳过
            "chr1\t500\t.\tC\tG,TT\t.\t.\t.\tGT\t0/1\t0/2\t0/0\t0/0",
            # 无 ALT 单调位点：计入 SNV 位点数，不产生候选
            "chr1\t600\t.\tA\t.\t.\t.\t.\tGT\t0/0\t0/0\t0/0\t0/0",
            # 全部缺失：AN=0，不产生候选
            "chr1\t700\t.\tA\tC\t.\t.\t.\tGT\t./.\t./.\t./.\t./.",
        ],
    )
    return path


# --------------------------------------------------------------------------- #
# 区间解析与合并
# --------------------------------------------------------------------------- #
def test_parse_intervals_valid_and_digit_strings():
    intervals = parse_intervals(
        [
            {"gene": " BRCA1 ", "chrom": "chr1", "start": "100", "end": "200"},
            {"gene": "TP53", "chrom": "chr2", "start": 1, "end": 1},
        ]
    )
    assert intervals[0] == Interval("BRCA1", "chr1", 100, 200)
    assert intervals[1] == Interval("TP53", "chr2", 1, 1)


def test_parse_intervals_rejects_bad_input():
    with pytest.raises(AnalysisError, match="至少"):
        parse_intervals([])
    with pytest.raises(AnalysisError, match="最多 10 条"):
        parse_intervals(
            [{"gene": f"G{i}", "chrom": "chr1", "start": 1, "end": 2}
             for i in range(11)]
        )
    with pytest.raises(AnalysisError, match="缺少基因名"):
        parse_intervals([{"gene": " ", "chrom": "chr1", "start": 1, "end": 2}])
    with pytest.raises(AnalysisError, match="缺少染色体"):
        parse_intervals([{"gene": "BRCA1", "chrom": "", "start": 1, "end": 2}])
    with pytest.raises(AnalysisError, match="超出允许范围"):
        parse_intervals([{"gene": "BRCA1", "chrom": "chr1", "start": 0, "end": 2}])
    with pytest.raises(AnalysisError, match="正整数"):
        parse_intervals(
            [{"gene": "BRCA1", "chrom": "chr1", "start": "abc", "end": 2}]
        )
    with pytest.raises(AnalysisError, match="1 基闭区间"):
        parse_intervals(
            [{"gene": "BRCA1", "chrom": "chr1", "start": 200, "end": 100}]
        )


def test_merge_intervals_same_gene_overlap():
    merged = merge_intervals_by_gene(
        [
            Interval("G1", "chr1", 1, 120),
            Interval("G1", "chr1", 100, 300),
            Interval("G1", "chr1", 500, 600),
            Interval("G1", "chr2", 1, 50),
            Interval("G2", "chr1", 1, 120),
        ]
    )
    assert merged["G1"]["chr1"] == [(1, 300), (500, 600)]
    assert merged["G1"]["chr2"] == [(1, 50)]
    assert merged["G2"]["chr1"] == [(1, 120)]


# --------------------------------------------------------------------------- #
# run_analysis 统计语义
# --------------------------------------------------------------------------- #
def test_low_freq_counts_and_missing_never_homref(unit_vcf):
    result = run_analysis(
        str(unit_vcf), [Interval("GENE1", "chr1", 1, 1000)]
    )
    assert result["samples"] == ["W", "X", "Y", "Z"]
    gene = result["genes"][0]
    assert gene["gene"] == "GENE1"
    assert gene["snv_sites"] == 7  # 100,150,200,250,300,600,700
    assert gene["skipped_non_snv"] == 2  # 400 (REF len2), 500 (ALT len2)
    assert gene["candidate_sites"] == 2  # 100, 150

    alleles = {(a["pos"], a["alt"]): (a["ac"], a["an"]) for a in gene["alleles"]}
    assert alleles == {(100, "G"): (1, 6), (150, "T"): (1, 4)}
    # 关键：chr1:250 若把缺失 GT 当 0/0，AC/AN 会变成 2/8=0.25 被纳入。
    assert 250 not in {a["pos"] for a in gene["alleles"]}

    per = gene["per_sample"]
    # W：chr1:100 有 GT 且带 1 份 G；chr1:150 缺失。
    assert per["W"] == {"alt_copies": 1, "called_sites": 1, "computable": True}
    # X：两个候选位点全部缺失 -> 无法计算。
    assert per["X"] == {"alt_copies": 0, "called_sites": 0, "computable": False}
    # Y：两位点均有 GT，仅在 chr1:150 带 1 份 T。
    assert per["Y"] == {"alt_copies": 1, "called_sites": 2, "computable": True}
    # Z：两位点均有 GT，均为 0/0。
    assert per["Z"] == {"alt_copies": 0, "called_sites": 2, "computable": True}


def test_overlapping_same_gene_intervals_count_site_once(unit_vcf):
    result = run_analysis(
        str(unit_vcf),
        [
            Interval("GENE1", "chr1", 1, 120),
            Interval("GENE1", "chr1", 100, 300),  # 与上一条重叠
            Interval("GENE2", "chr1", 1, 120),  # 不同基因，独立统计
        ],
    )
    g1, g2 = result["genes"]
    assert g1["intervals"] == [{"chrom": "chr1", "start": 1, "end": 300}]
    # chr1:100 被两条区间覆盖，但只算一次。
    assert g1["candidate_sites"] == 2
    assert g1["snv_sites"] == 5
    assert g2["candidate_sites"] == 1
    assert g2["snv_sites"] == 1


def test_multiallelic_both_alts_qualify_and_copies_sum(tmp_path):
    path = tmp_path / "multi.vcf"
    vcf_with(
        path,
        ["W", "X", "Y", "Z"],
        ["chr1\t100\t.\tA\tG,T\t.\t.\t.\tGT\t1/2\t./.\t./.\t0/0"],
    )
    result = run_analysis(str(path), [Interval("GENE1", "chr1", 1, 1000)])
    gene = result["genes"][0]
    # AN=4（W、Z 可判），G 与 T 各 AC=1 -> 0.25，两个 ALT 都入选。
    assert {(a["alt"], a["ac"], a["an"]) for a in gene["alleles"]} == {
        ("G", 1, 4),
        ("T", 1, 4),
    }
    per = gene["per_sample"]
    # W 的 GT=1/2，两个低频 ALT 各一份 -> 拷贝数 2。
    assert per["W"]["alt_copies"] == 2
    assert per["W"]["called_sites"] == 1
    assert per["Z"]["alt_copies"] == 0
    assert per["X"]["computable"] is False


def test_run_analysis_aborts_when_invalidated(unit_vcf):
    with pytest.raises(AnalysisAborted):
        run_analysis(
            str(unit_vcf),
            [Interval("GENE1", "chr1", 1, 1000)],
            is_aborted=lambda: True,
        )


# --------------------------------------------------------------------------- #
# API 工作流
# --------------------------------------------------------------------------- #
def wait_for_analysis(client, batch_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/batches/{batch_id}/analyses/latest").json()
        analysis = payload["analysis"]
        if analysis and analysis["status"] in ("done", "failed", "superseded"):
            return analysis
        time.sleep(0.05)
    raise AssertionError("analysis did not reach a terminal state")


def prepare_cohort(client):
    """上传三份演示文件、选定 S2/S3 取用源并生成队列 VCF。"""
    data = upload_files(client, ["batch_a.vcf", "batch_b.vcf", "batch_c.vcf"]).json()
    batch_id = data["id"]
    dup = {g["sample_name"]: g for g in data["dup_groups"] if g["is_duplicate"]}
    for sample, part in (("S2", "batch_a"), ("S3", "batch_c")):
        group = dup[sample]
        choice = next(
            c["file_id"] for c in group["choices"] if part in c["file_name"]
        )
        resp = client.put(
            f"/api/batches/{batch_id}/selection/{group['id']}",
            data={"file_id": choice},
        )
        assert resp.status_code == 200, resp.text
    assert client.post(f"/api/batches/{batch_id}/generate").status_code == 200
    final = wait_for_terminal(client, batch_id)
    assert final["latest_generation"]["status"] == "done"
    return batch_id


def post_analysis(client, batch_id, intervals):
    return client.post(
        f"/api/batches/{batch_id}/analyses", json={"intervals": intervals}
    )


def test_analysis_api_workflow(client):
    batch_id = prepare_cohort(client)

    resp = post_analysis(
        client, batch_id, [{"gene": "GENE1", "chrom": "chr1", "start": 1, "end": 500}]
    )
    assert resp.status_code == 200, resp.text
    analysis = wait_for_analysis(client, batch_id)
    assert analysis["status"] == "done", analysis["error"]
    assert analysis["is_current"] is True
    assert analysis["stale_reason"] is None

    result = analysis["result"]
    assert sorted(result["samples"]) == ["S1", "S2", "S3", "S4"]
    gene = result["genes"][0]
    assert gene["snv_sites"] == 3  # chr1:100, chr1:250, chr1:400
    assert gene["candidate_sites"] == 1
    # 队列 chr1:100：G 的 AC=3/AN=8=0.375 排除；T 的 AC=1/AN=8=0.125 入选。
    assert gene["alleles"] == [
        {
            "chrom": "chr1", "pos": 100, "ref": "A", "alt": "T",
            "ac": 1, "an": 8, "af": 0.125,
        }
    ]
    per = gene["per_sample"]
    assert per["S1"] == {"alt_copies": 0, "called_sites": 1, "computable": True}
    assert per["S2"] == {"alt_copies": 1, "called_sites": 1, "computable": True}
    assert per["S3"] == {"alt_copies": 0, "called_sites": 1, "computable": True}
    assert per["S4"] == {"alt_copies": 0, "called_sites": 1, "computable": True}


def test_analysis_requires_completed_cohort(client):
    data = upload_files(client, ["batch_a.vcf", "batch_b.vcf"]).json()
    batch_id = data["id"]
    resp = post_analysis(
        client, batch_id, [{"gene": "G1", "chrom": "chr1", "start": 1, "end": 10}]
    )
    assert resp.status_code == 409
    assert "请先生成" in resp.json()["detail"]


def test_analysis_interval_validation_via_api(client):
    batch_id = prepare_cohort(client)
    too_many = post_analysis(
        client,
        batch_id,
        [{"gene": f"G{i}", "chrom": "chr1", "start": 1, "end": 2} for i in range(11)],
    )
    assert too_many.status_code == 400
    assert "最多 10 条" in too_many.json()["detail"]

    bad_range = post_analysis(
        client, batch_id, [{"gene": "G1", "chrom": "chr1", "start": 50, "end": 10}]
    )
    assert bad_range.status_code == 400
    assert "1 基闭区间" in bad_range.json()["detail"]

    missing_gene = post_analysis(
        client, batch_id, [{"gene": "", "chrom": "chr1", "start": 1, "end": 10}]
    )
    assert missing_gene.status_code == 400


def test_selection_change_invalidates_analysis(client):
    batch_id = prepare_cohort(client)
    assert post_analysis(
        client, batch_id, [{"gene": "G1", "chrom": "chr1", "start": 1, "end": 500}]
    ).status_code == 200
    analysis = wait_for_analysis(client, batch_id)
    assert analysis["status"] == "done" and analysis["is_current"] is True

    # 变更取用源：旧分析立即失效，且结果表不再下发。
    data = client.get(f"/api/batches/{batch_id}").json()
    group = next(g for g in data["dup_groups"] if g["sample_name"] == "S2")
    other = next(
        c["file_id"] for c in group["choices"] if "batch_b" in c["file_name"]
    )
    client.put(
        f"/api/batches/{batch_id}/selection/{group['id']}",
        data={"file_id": other},
    )
    stale = client.get(f"/api/batches/{batch_id}/analyses/latest").json()["analysis"]
    assert stale["status"] == "superseded"
    assert stale["is_current"] is False
    assert stale["result"] is None  # 旧表不可继续显示为当前结果

    # 重新生成后可以重新分析，新结果为当前结果。
    assert client.post(f"/api/batches/{batch_id}/generate").status_code == 200
    final = wait_for_terminal(client, batch_id)
    assert final["latest_generation"]["status"] == "done"
    assert post_analysis(
        client, batch_id, [{"gene": "G1", "chrom": "chr1", "start": 1, "end": 500}]
    ).status_code == 200
    fresh = wait_for_analysis(client, batch_id)
    assert fresh["status"] == "done"
    assert fresh["is_current"] is True
    assert fresh["result"] is not None


def test_regeneration_invalidates_analysis(client):
    batch_id = prepare_cohort(client)
    assert post_analysis(
        client, batch_id, [{"gene": "G1", "chrom": "chr1", "start": 1, "end": 500}]
    ).status_code == 200
    analysis = wait_for_analysis(client, batch_id)
    assert analysis["status"] == "done"

    # 队列结果被重新生成：旧分析立即失效。
    assert client.post(f"/api/batches/{batch_id}/generate").status_code == 200
    wait_for_terminal(client, batch_id)
    stale = client.get(f"/api/batches/{batch_id}/analyses/latest").json()["analysis"]
    assert stale["status"] == "superseded"
    assert stale["is_current"] is False
    assert stale["result"] is None


def test_concurrent_analysis_rejected(client, app_env):
    batch_id = prepare_cohort(client)
    db = app_env["db"]
    config = app_env["config"]
    with db.get_db() as conn:
        batch = db.get_batch(conn, batch_id)
        gen = db.get_latest_generation(conn, batch_id)
        analysis_id = db.create_analysis(
            conn, batch_id, gen["id"], batch["selection_version"],
            [{"gene": "G1", "chrom": "chr1", "start": 1, "end": 2}],
        )
        db.mark_analysis(conn, analysis_id, config.STATUS_RUNNING, started=True)
    resp = post_analysis(
        client, batch_id, [{"gene": "G2", "chrom": "chr1", "start": 1, "end": 10}]
    )
    assert resp.status_code == 409
    assert "进行中" in resp.json()["detail"]


def test_restart_marks_running_analysis_failed(client, app_env):
    batch_id = prepare_cohort(client)
    db = app_env["db"]
    config = app_env["config"]
    with db.get_db() as conn:
        batch = db.get_batch(conn, batch_id)
        gen = db.get_latest_generation(conn, batch_id)
        analysis_id = db.create_analysis(
            conn, batch_id, gen["id"], batch["selection_version"],
            [{"gene": "G1", "chrom": "chr1", "start": 1, "end": 10}],
        )
        db.mark_analysis(conn, analysis_id, config.STATUS_RUNNING, started=True)

    db.init_db()  # 模拟服务重启
    analysis = client.get(f"/api/batches/{batch_id}/analyses/latest").json()["analysis"]
    assert analysis["status"] == "failed"
    assert "重启" in analysis["error"]
    assert analysis["result"] is None
