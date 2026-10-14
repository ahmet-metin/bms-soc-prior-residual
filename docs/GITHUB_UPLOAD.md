# GitHub upload — short checklist

The ZIP root is the repository folder. Extract it and upload **the files inside the root** to a new GitHub repository, or push them from a local Git clone.

1. Create a GitHub repository (suggested name: `bms-soc-prior-residual`) and **do not** auto-add a second README / `.gitignore`.
2. Extract the ZIP. Before publishing, confirm that `LICENSE_STATUS.md`, `README.md`, `docs/VERIFICATION_STATUS.md` and `docs/SOURCE_PROVENANCE.md` accurately describe what you wish to disclose.
3. From the extracted repository root, run:

   ```bash
   python -m pip install -e '.[all]'
   python -m unittest discover -s tests -v
   python scripts/verify_sources.py
   python scripts/export_notebooks_to_py.py --check
   python scripts/check_archived_aulec.py
   ```

4. Scan any manually added files for secrets, raw telemetry, vehicle identifiers, checkpoints, private links and data-use restrictions. Generated outputs, raw datasets and model weights are deliberately omitted from this ZIP.
5. Optional Git CLI upload (replace the URL with your actual repository URL):

   ```bash
   git init
   git add .
   git commit -m "Initial research code release"
   git branch -M main
   git remote add origin https://github.com/YOUR-USER/YOUR-REPO.git
   git push -u origin main
   ```

6. Do not claim the repository establishes independent 350/350 retraining, unseen-vehicle transfer or novel independent EVE confirmation. The code may be published transparently before those claims are verified.
7. **Licensing is an author decision.** This package does not assert an open-source license; obtain necessary permissions before choosing one.

Do not paste a fabricated GitHub URL or DOI into the manuscript. Add the actual URL after the repository exists. Record a DOI only after an archive service issues it.
