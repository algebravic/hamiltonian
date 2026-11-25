import sys
for f in sys.argv[1:]:
    W=0; pk=0; t=None
    for line in open(f):
        s=line.split()
        if len(s)>=10 and s[0].isdigit() and s[1].isdigit():
            W+=int(s[5])*2**int(s[3]); pk=max(pk,int(s[6]))
        if line.startswith("RESULT"): t=line.split("dp=")[1].split()[0]
    print(f"{f}: W={W:.4g} peak={pk:.4g} dp={t}")
