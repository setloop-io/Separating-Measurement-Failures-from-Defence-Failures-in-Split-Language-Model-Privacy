"""Input admission regressions: no empty, reused or cross-group windows."""
import pytest
from experiments.freeze_bisr_fresh_inputs import choose
from experiments.freeze_bisr_inputs import text_hash
from attacker.bisr_fidelity import compare_frozen_scores


def doc(name, text):
    return {'id':name,'title':name,'text':text,'normalised_sha256':text_hash(text)}


def test_selection_rejects_history_empty_and_validation_windows():
    rows=[doc('historical','history'),doc('short','short'),doc('copy','copy'),doc('valid','valid')]
    tokens={'history':list(range(32)), 'short':[1,2], 'copy':[7]*32, 'valid':[9]*64}
    selected,rejected=choose(rows,1,'salt',lambda t:tokens[t],
        {text_hash('history')},set(),{tuple([7]*32)},True)
    assert [r['id'] for r in selected]==['valid']
    # Insufficient valid inputs must never degrade to empty articles or fewer rows.
    with pytest.raises(ValueError,match='no reduced-count fallback'):
        choose(rows,2,'salt',lambda t:tokens[t],{text_hash('history')},set(),{tuple([7]*32)},True)


def test_articles_cannot_be_duplicated_by_title_or_text():
    rows=[doc('a','same'),doc('b','same')]
    with pytest.raises(ValueError):
        choose(rows,2,'salt',lambda t:list(range(32)),set(),set(),set())
    with pytest.raises(ValueError):
        choose([doc('Excluded Title','body')],1,'salt',lambda t:list(range(32)),
               set(),{'excluded title'},set())


def test_fresh_design_does_not_accept_reduced_score_count():
    with pytest.raises(ValueError,match='all 256'):
        compare_frozen_scores({}, {}, {'schema':'bisr.frozen_inputs.v3',
                                      'validation_ids':['a'],'validation_documents':1})
