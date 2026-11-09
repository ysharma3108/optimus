# Verified examples

Scripts the assistant generated that **ran successfully in the OptimUS Docker
image**. The router (route 2) adapts these instead of writing new code.

Rules (hybrid design of 3 Nov 2026):

1. A script is only added here after it passed the checks and the Docker run.
2. Each file starts with a docstring whose first line is
   `Request: <the request it answers>`, followed by the date, the commit it was
   verified on and the run time.
3. CI re-runs these scripts; one that stops working is removed or fixed.
4. Yash reviews a script before it is promoted into a curated notebook.

The folder is empty until the pilot produces the first verified scripts.
