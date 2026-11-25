# Paper B submission handoff

Checked **15 October 2026**. These are submission candidates; **neither version has been submitted to arXiv or EuroS&P**. The repository/evidence release is a separate operation, tracked in the [v1.0 release handoff](v1.0-release-handoff.md). Build checks do not establish scientific acceptance or complete the remaining author decisions.

## Current candidates

- [Anonymous EuroS&P PDF](../output/pdf/paper-b-publication/paper-b-eurosp2027-anonymous.pdf)
- [Combined concise-plus-supplement arXiv PDF](../output/pdf/paper-b-publication/paper-b-arxiv-named.pdf)
- [Named arXiv source ZIP](../output/pdf/paper-b-publication/paper-b-arxiv-named-source.zip)
- [arXiv metadata](../output/pdf/paper-b-publication/arxiv-metadata.json)
- [Build verification](../output/pdf/paper-b-publication/build-verification.json)
- [Publication configuration and outstanding confirmations](../paper/paper_b/publication.json)

The conference candidate has 25 pages, with its body ending on page 7. The combined arXiv candidate has 41 pages: the concise paper ends on page 9, supplementary material starts on page 10, and a single bibliography starts on page 40. Both use A4 pages and embedded fonts, have resolved references, and compile successfully from their extracted source archives. Both candidates remain marked `release_ready: false`.

## IEEE EuroS&P 2027

The [official HotCRP portal](https://hotcrp.eurosp2027.ieee-security.org/) displayed **“Submissions are currently closed.”** No opening date is stated in the [official CFP](https://eurosp2027.ieee-security.org/cfp.html). Prepare now and register when the portal opens; the deadlines are not opening dates.

| Stage | Official deadline, AoE (UTC−12) | End-of-day equivalent in London (GMT) |
| --- | --- | --- |
| Mandatory abstract registration | 25 November 2026 | Before noon, 26 November 2026 |
| Paper submission | 2 December 2026 | Before noon, 3 December 2026 |

The London conversions assume the end of the stated AoE day. Use an earlier internal deadline and recheck the portal's exact closing time. Notification is scheduled for 18 March 2027; camera-ready is tentatively 15 April 2027.

The CFP requires an anonymous PDF, at most **13 body pages**, with unlimited references and appendices, using its [specific LaTeX A4 template](https://eurosp2027.ieee-security.org/eurosp2027-template.zip). Remove identifying author details and acknowledgements; self-cite in third person. Hosted review materials must be anonymous: ordinary GitHub and Google Drive are explicitly excluded. An arXiv preprint is allowed during review. Substantially similar simultaneous conference/journal submissions are prohibited, and overlap with related author papers must be explained. Discuss potential harms in the body. AI-generated content requires an acknowledgement identifying the system, affected sections and extent of use; grammar-only assistance is exempt, though disclosure is recommended. Registration needs title, abstract, authors, conflicts, primary area and topics. See the [CFP](https://eurosp2027.ieee-security.org/cfp.html) for complete requirements.

Use the web portal and anonymous PDF when submissions open. No conference-specific API submission workflow is documented by the CFP.

## arXiv

arXiv can proceed before EuroS&P opens. Start from the [author dashboard](https://arxiv.org/user), then **START NEW SUBMISSION**. The browser currently reaches the login screen; account and endorsement eligibility have not been established. A new submitter or category may require [endorsement](https://info.arxiv.org/help/endorsement.html).

Upload the named source ZIP, review the server-compiled PDF, then check the metadata. arXiv requires source for a paper written in TeX; uploading only the locally generated PDF is insufficient. The archive must contain its required figures, macros/styles and bibliography, without unrelated files. See the [submission guide](https://info.arxiv.org/help/submit/index.html) and [TeX requirements](https://info.arxiv.org/help/submit_tex.html).

The documented [public API](https://info.arxiv.org/help/api/basics.html) serves content and metadata retrieval. No general public author-upload API was identified; use the web author workflow. Suggested primary category: **`cs.CR`**, based on the [category descriptions](https://arxiv.org/category_taxonomy); confirm this choice in the form.

The selected [arXiv distribution licence](https://info.arxiv.org/help/license/index.html) is the arXiv.org perpetual, non-exclusive licence, confirmed on 15 October 2026. The grant for each posted version is irrevocable. Allow time for moderation and checks, typically one to four days and sometimes longer. Announcements normally run Sunday–Thursday at 20:00 US Eastern under the [published schedule](https://info.arxiv.org/help/availability.html).

## Remaining work before submission

1. Author details are retained as confirmed: Georgios Politis, then Evangelos Pappas, with existing email addresses. The requested arXiv title layout displays Setloop.io between the two author blocks and uses linked monospace emails. Obtain all authors' agreement to the final paper and submission details.
2. Confirm prior AI systems and tasks, then complete the acknowledgements accurately. Confirm the status and overlap of related papers, including Paper A; disclose any material overlap as required.
3. Arrange and test anonymous reviewer access to the necessary code and complete evidence. No verified anonymous artifact URL is configured yet.
4. Use the [named Hugging Face v1.0 release pointer](https://huggingface.co/datasets/Setloop/The-LLM-Attacker-Unified-Red-Team-Framework-for-Privacy-Evaluation-of-Split-and-Distributed-LLMs/tree/v1.0), verified at immutable revision [`a0df3938e993b2d532904459fd399c4dab2d1301`](https://huggingface.co/datasets/Setloop/The-LLM-Attacker-Unified-Red-Team-Framework-for-Privacy-Evaluation-of-Split-and-Distributed-LLMs/tree/a0df3938e993b2d532904459fd399c4dab2d1301). The [release handoff](v1.0-release-handoff.md) links the completed GitHub/Hugging Face history and tag verification receipts. The dataset listing is public, protected file downloads require a Hugging Face account and **manual approval**, and the source repository requires authorised GitHub access. This named release does not supply anonymous review access.
5. Use the selected arXiv non-exclusive licence, establish account/endorsement access and confirm the category. Update the publication configuration as the remaining issues are resolved, rebuild, and review both final candidates and their verification report.
6. Submit the named source through arXiv's web workflow after reviewing its preview and terms. Once HotCRP opens, complete mandatory registration and upload the anonymous PDF before the respective deadlines. Recheck the official instructions at submission time.

This guide does not submit to arXiv or HotCRP. Repository/evidence publication and its verification receipts are tracked separately; they do not establish author-dashboard access, endorsement, registration or final paper submission.

The existing `build/reviewer-repos/paper-b/` is a historical named reproduction package, not an anonymous upload candidate: it contains identifying manuscript details and named repository/data links. The anonymous manuscript source ZIP contains paper sources only. A separate reviewer evidence package must preserve the pinned experimental contents while removing identifying presentation metadata, and must be tested without author-account access before its URL is inserted into the conference PDF.
