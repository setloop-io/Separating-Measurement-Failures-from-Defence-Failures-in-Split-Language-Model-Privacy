# How we write a paper: the Paper A playbook

This guide records what made Paper A ("What Gradients Add to Text Leakage in Split Language Models, Counted per Token and per Document", September 2026) read well, and the process that got it there. Use it for Paper B and later papers.

The quality bar came from six arXiv papers the authors chose (2609.28613, 2609.29429, 2609.26550, 2609.30266, 2609.31379) and above all 2609.31142 (JevAdvBench) for voice and Figure 1.

## 1. The core idea comes first

- One sentence states the paper's idea before any method: "How much private text leaks from a split language model depends as much on how leakage is counted as on what the attacker sees."
- The title asks the paper's question in plain words. It does not list nouns and does not use our shorthand. Rejected: "Scoring Text Leakage in Split Language Models: Tokens, Whole Documents, and the Returned Gradient". Accepted: "What Gradients Add to Text Leakage in Split Language Models, Counted per Token and per Document".
- The paper must be clearly distinct from our earlier work. Cite our own earlier papers where we build on them, say in one sentence what the earlier paper found, and reconcile any number that seems to disagree with it.

## 2. Structure of the document

Abstract, about 200-240 words, in this order:
1. The setting in plain words (who sends what to whom).
2. What we measure.
3. The main result, with the headline numbers next to their baseline.
4. Why the result matters ("Counted by document, the difference is much larger").
5. The second finding (defences).
6. The second experiment in one sentence.
7. One recommendation a practitioner can act on.

Introduction, one job per paragraph, each paragraph opening from the previous one:
1. The setting: what the system is, what crosses between the parties, why that is a risk. Each technical term (activation, gradient) is explained once, in plain words, in italics.
2. What earlier work shows, each prior work explained in one clause before we relate to it, then the gap: what this does not tell a practitioner.
3. Our questions, asked directly.
4. What we found for the first question, pointing to Figure 1.
5. The second finding.
6. What we found for the second question.
7. Contributions: four bullets, each opening with a bold plain label ("How to score defences.", "What gradients add.", "Where offloading starts.", "Everything released.") followed by full sentences with a subject ("We show...", "We measure...", "We release...").

Body:
- Related work: one plain clause per prior work saying what it does, then how ours differs.
- Threat model and setup opens with an architecture diagram. It says who the attacker is, what it holds and what it never gets.
- A section called "How the attack works" walks through the method in numbered steps a reader can picture, matched one-to-one to a method diagram.
- A table of the signals the attacker reads: signal, when it crosses, how the attack uses it, what it reveals.
- Results are organised by finding; headings state findings in plain words.
- Scope and limitations: short, stated as fact, each limitation with its consequence (see section 4).
- Conclusion: the main claim, then concrete recommendations.
- Appendix for the release: plain paragraphs on what is released, where, and how to reproduce it. All hashes, revisions and file names go in one table at the end, never inside sentences.

## 3. Voice and wording

- Write like a careful human researcher. A person or thing does something with a verb: "the attacker recovers", "we measured", "the defence costs". Avoid noun chains with weak verbs ("is", "has", "gives").
- Each sentence follows from the one before. Use the link a human would (because, so, but, then, for example, by contrast) where the logic needs it.
- Short declaratives carry the argument; statistics come after the plain statement. Numbers always sit next to their baseline.
- Explain every term once, at first use, in plain words, then use the same word every time. One name per thing (we had five names for "whole-document recovery"; that confused readers).
- Never use project shorthand in the paper: internal stage names (v3, P6, P0-P5), "campaign", "cells", "arms", "grants", "endpoints", "confirmatory", "prospective", "head" (to ML readers it means the output layer), "channel", "blocks" when we mean layers.
- State each caveat once, where the reader first needs it.
- Say which results were tested against a bar fixed in advance and which are descriptive, once per section, in plain words.

Sentences the authors approved, as models:
- "That looks like a small gain until we count whole documents."
- "One wrong token is enough to spoil a document, so fixing the last few errors turns many near misses into exact copies."
- "Anyone who judged this defence only by exact copies would think it much stronger than it is."
- "An attacker who knows the defence can always run these attacks, so our defended results show what such an attacker recovers at the least."

Phrases the authors rejected as signs of machine writing:
- "tell a different story", "the other way", "the answer to the first question", "in short", "put differently", "taken together", "crucially", "notably"
- unnatural "a single X" ("fails on a single wrong token")
- colon reveals ("The result: ..."), "X, not Y" slogans, triads of parallel clauses for rhythm
- definitions stacked in asides ("X, that is, Y", "X, the Y of Z,")
- formulaic openers and closers ("This section answers...", "Four recommendations follow")
- sentences that restate the previous sentence

## 4. Tone

- No hedging and no defensive wording. The measured data are the source of truth.
- No hype either. Claims stay within what the numbers show; descriptive results are not presented as tested ones.
- Limitations are stated as fact with their consequence, not as apology. Example: "An attacker who knows the defence can always run these attacks, so our defended results show what such an attacker recovers at the least; attacks designed around a defence can only add to it."

## 5. Figures and tables

Figure 1 is the "gasp" figure: top of page 2, full text width, three panels, readable in three seconds.
- (a) The setting: what leaves the client, the split, what comes back, where the attacker sits and what it holds.
- (b) One real example from the released data: the true text next to both attackers' reconstructions, wrong tokens highlighted, invisible errors (a leading space) marked. The example is chosen by a stated rule (closest to the reported averages), never by eye.
- (c) The aggregate result with intervals, a grey baseline band, and the one ratio that matters called out large (2.75x).
- The caption opens with a bold one-line takeaway in plain words, then says what each panel shows and how the example was chosen.

Other figures:
- An architecture diagram at the start of the setup section: client, split, server, attacker; what it holds and what it never gets; an inset for the second experiment.
- A method diagram whose numbered badges match the numbered steps in the text; controls drawn as dashed side branches.
- One palette across all figures, muted colours with one accent for errors, direct labels instead of legends, text 8 pt or larger at print size, the same terms as the text.
- Every plotted number is computed from the released data by a script in the repository, which checks that it rounds to the value in the text and stops otherwise. Schematic parts must not look like data.

Tables: booktabs, ragged-right text columns, spacing between rows, 2-3 significant digits in the main text, full precision in the appendix.

## 6. Authorship and positioning

- Evangelos Pappas: name and email only, no affiliation.
- Cite our own earlier papers where we build on them, and position against them explicitly.
- Every method explanation must come from the project's own documentation or code (cited to file and line in the working notes); nothing is invented. Where the paper and the documentation disagree, the documentation wins and the fix is listed for the co-author to confirm.
- Our co-author's data is the source of truth for numbers.

## 7. The process that worked

- One author writes the voice anchor. Agent-written prose failed four times; the abstract and introduction drafted by one author in a single sitting were accepted at once ("those drafts make more sense"). Draft the abstract and introduction first, get the co-authors' approval, then bring every other section to that voice.
- Keep data frozen by script: after every edit, compare the set of numbers, citation keys and labels with the previous version, and rebuild from an empty directory (pdflatex, bibtex, pdflatex, pdflatex): 0 errors, 0 undefined references, 0 overfull boxes.
- Look at the rendered pages, not only the source. Figure placement and table typography only show in the PDF.
- Measure claims yourself before relaying them. An agent reported that first-token errors were "almost always" a quotation mark; measured, it was one third. The paper now reports the measured share.
- Readers with no context (an ML researcher, a security engineer) are useful for finding recurring problems: undefined terms, missing mechanism, jargon. Their flag counts are not a quality score: an exemplar paper scored 65 flags on the same test. The authors' own reading decides.
- Check for overlap with our earlier papers before release (8-word shared runs; ours was 0%), and check for positioning, not just text reuse.
- Release: one commit on Hugging Face that adds a dated manuscript folder (PDF, LaTeX source zip that builds from an empty directory, figures, figure scripts, a SOURCE.txt with the git commit), replaces the stale copy, points README and INDEX to it, and updates MANIFEST.sha256 so readers' checksum checks still pass. Download it back and verify against the manifest.
