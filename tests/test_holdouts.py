from samileides.holdouts import propose_extended_holdouts


def test_proposal_one_per_bucket(synthetic_metadata, monkeypatch):
    monkeypatch.setattr(
        "samileides.holdouts.load_families",
        lambda: {
            "swh": "Bantu",
            "ceb": "Austronesian",
            "hin": "Indo-Aryan",
            "tur": "Turkic",
            "vie": "Austroasiatic",
            "eng": "Indo-European",
        },
    )
    proposal = propose_extended_holdouts(synthetic_metadata)
    families = list(proposal["family"])
    assert families == ["Bantu", "Austronesian", "Indo-Aryan", "Turkic", "Austroasiatic"]
    # English is an existing holdout and must never be proposed
    assert "eng" not in set(proposal["languageCode"])
    assert set(proposal["heldOutBook"]) == {"GEN"}
