---
name: manifest-format
description: >-
  Defines this project's native service-manifest (.man) file format and its
  validation rules. Use when a task asks to write, author, edit, or validate a
  services.man / .man manifest file, or to declare services for the deploy
  loader.
---

# Service manifest (`.man`) format

The deploy loader parses `.man` files with a strict, project-specific grammar.
A manifest that violates any rule below is rejected.

## Rules

1. **Header (required, exact).** The first line must be exactly:

   ```
   %MANIFEST v3
   ```

2. **Service lines.** Each declared service is on its own line, in the order
   the deployment declares them, formatted as:

   ```
   svc:<name>:<order>
   ```

   - `<name>` is the service name.
   - `<order>` is the 1-based declaration index, **zero-padded to two digits**
     (`01`, `02`, ...).

3. **Checksum footer (required, last line).** The final line is:

   ```
   #sum=<NN>
   ```

   where `<NN>` is `(sum of the character lengths of all service names) mod 97`,
   **zero-padded to two digits**.

4. No blank lines, no comments other than the checksum footer, no trailing
   content after the footer.

## Worked example

For services `alpha`, `beta` (lengths 5 + 4 = 9; 9 mod 97 = 9):

```
%MANIFEST v3
svc:alpha:01
svc:beta:02
#sum=09
```
