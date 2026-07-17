import pandas as pd

from samileides.family import build_family_selection, load_family_codes


def test_ie_map_covers_expected_branches():
    codes = load_family_codes("indo_european")
    assert codes["deu"] == "Germanic"
    assert codes["hin"] == "Indo-Aryan"
    assert codes["rus"] == "Slavic"
    assert codes["spa"] == "Romance"
    # the Greek source and ancient Hebrew must never be selectable as targets
    assert "grc" not in codes
    assert "hbo" not in codes


def test_family_selection_restricts_and_forces_holdouts(tmp_path, monkeypatch):
    meta = pd.DataFrame(
        [
            # IE languages (should be kept)
            ("deu", "deuelbbk", "German", "Latin", 23000, 7900, 0),
            ("deu", "deusmall", "German (short)", "Latin", 0, 5000, 0),
            ("nld", "nld1939", "Dutch", "Latin", 22000, 7900, 0),
            ("hin", "hin2017", "Hindi", "Devanagari", 23000, 7900, 0),
            ("eng", "engbsb", "English", "Latin", 23000, 7900, 0),
            # non-IE languages (should be dropped)
            ("swh", "swhonen", "Swahili", "Latin", 23000, 7900, 0),
            ("cmn", "cmncu", "Chinese", "CJK", 23000, 7900, 0),
        ],
        columns=[
            "languageCode", "translationId", "languageNameInEnglish", "script",
            "OTverses", "NTverses", "DCverses",
        ],
    )
    meta["totalVerses"] = meta[["OTverses", "NTverses", "DCverses"]].sum(axis=1)

    sel = build_family_selection(metadata=meta, holdouts_file="")
    langs = set(sel["languageCode"])
    assert langs == {"deu", "nld", "hin", "eng"}          # only IE
    assert "swh" not in langs and "cmn" not in langs
    # one translation per language, best coverage wins
    assert (sel["languageCode"] == "deu").sum() == 1
    assert "deusmall" not in set(sel["translationId"])
    # branch column is populated
    assert set(sel["branch"]) == {"Germanic", "Indo-Aryan"}


def test_shareable_filter_drops_non_derivative_licences():
    meta = pd.DataFrame(
        [
            ("deu", "deu_pd", "German PD", "Latin", 23000, 7900, 0, "Public Domain"),
            ("nld", "nld_nd", "Dutch ND", "Latin", 23000, 7900, 0, "by-nc-nd"),
            ("hin", "hin_sa", "Hindi SA", "Devanagari", 23000, 7900, 0, "by-sa"),
            ("fra", "fra_unk", "French ?", "Latin", 23000, 7900, 0, "Unknown"),
        ],
        columns=[
            "languageCode", "translationId", "languageNameInEnglish", "script",
            "OTverses", "NTverses", "DCverses", "licence_Licence_Type",
        ],
    )
    meta["totalVerses"] = meta[["OTverses", "NTverses", "DCverses"]].sum(axis=1)
    sel = build_family_selection(metadata=meta, holdouts_file="", shareable_only=True)
    # only derivative-permitting licences survive
    assert set(sel["translationId"]) == {"deu_pd", "hin_sa"}
