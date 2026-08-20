# Always Verify Before and After Implementing Changes

1. **Pre-Implementation Check**: Always run the test suite and verify current pipeline runs before modifying files.
2. **Post-Implementation Verification**: Always run `python -m unittest discover -s tests` and execute component validation checks after applying edits.
3. **Integrity Assurance**: Never consider a feature complete without automated test execution showing zero failures.
