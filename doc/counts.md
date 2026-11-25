Exact count of edges in the square graph
========================================

If \(n\) is a positive integer, define \(G(n)\) as the undirected
graph with \(n\) vertices labeled \(1,\dots,n\), and \(\{i,j\}\) is an
edge if and only if \(i+j\) is a perfect square.

Note that \(G(n)\) is an induced subgraph of \(G(n+1)\). The extra
edges in \(G(n+1)\) are \((n+1,j)\) where \(n+1+j\) is a perfect
square, and \(1 \le j \le n\).  If we write \(n+1+j=m^2\), the values
of \(j\) are in one-to-one correspondence with integers
\(m\), such that \(n+2 \le m^2 \le 2n+1\). Thus the number of edges in
\(G(n)\) is

\[\sum_{j=1}^n (\lfloor sqrt{2j-1}\rfloor - \lceil \sqrt{j+1} \rceil +
1) \]

Note that an integer \(m = \lfloor \sqrt{2j-1} \rfloor\) if and only
if \(\sqrt{2j-1} - 1 < m \le sqrt{2j-1}\) which is equivalent to
\(2j-1 < (m+1)^2\) and \(m^2 \le 2j-1\). Rewriting this gives
\((m^2+1)^2/2 \le j \le (m+1)^2/2\).

Thus
\[
\sum_{j=1}^n \lfloor \sqrt{2j-1} \rfloor =
\sum_{m=1}  m(\lfloor \frac{(m+1)^2}{2} \rfloor - \lceil \frac{m^2+1}{2}
\rceil + 1)
\]

If \(m\) is even, then \(\lfloor\frac{(m+1)^2}{2} \rfloor =
\frac{(m+1)^2-1}{2}\) and \(lceil\frac{m^2+1}{2}\rceil =
\frac{m^2+2}{2}\).

If \(m\) is odd, then \(\lfloor\frac{(m+1)^2}{2} \rfloor =
\frac{(m+1)^2}{2}\) and \(lceil\frac{m^2+1}{2}\rceil =
\frac{m^2+1}{2}\).

Or \((\lfloor \frac{(m+1)^2}{2} \rfloor - \lceil \frac{m^2+1}{2}
\rceil + 1) = m + 1 - \frac{1+(-1)^m}{4} \)
