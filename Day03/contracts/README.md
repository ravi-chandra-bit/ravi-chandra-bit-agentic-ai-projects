# Day 3 — Sample Contract Corpus

Six real, public contract documents fetched from the U.S. SEC EDGAR full-text
search system (filed as EX-10 material-contract exhibits). They serve as the
input library for `Day_03_contract_clause_search.ipynb` (RAG / FAISS semantic
search). All are public-domain regulatory filings. Retrieved 2026-06-16.

Each `.txt` file begins with a metadata header (filer, exhibit type, source URL).

| File | Filer | Exhibit | Clause types it exercises |
|------|-------|---------|---------------------------|
| `employment_noncompete_agreement.txt` | PJT Partners Inc. | EX-10.17 | non-compete, non-solicitation |
| `mutual_nda.txt` | International Paper Co. | EX-10.1 | confidentiality / non-disclosure |
| `ip_assignment_agreement.txt` | World Technology Corp. | EX-10.1 | IP ownership / assignment |
| `master_services_agreement.txt` | Sunrise Communications AG | EX-10.4 | services, SLAs, liability |
| `software_license_agreement.txt` | SS&C Technologies Inc. | EX-10.1 | license grant, IP, restrictions |
| `consulting_agreement.txt` | Marlin Business Services Corp. | EX-10.2 | consulting terms, confidentiality |

These cover the synonym-heavy queries the notebook tests, e.g. "restrictions on
hiring our employees after the contract ends" → non-solicitation language, and
"who owns the IP if we co-develop something" → IP assignment clauses.

Source: https://efts.sec.gov/LATEST/search-index (SEC EDGAR full-text search)
