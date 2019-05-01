# Impute

## Introduction
*Impute* implements a couple of known low-rank matrix imputing methods. Namely, you can
find the implementation of the following algorithms:
 1. **Fixed-point continuation method** (https://arxiv.org/abs/0905.1643)
 2. **Soft-impute method** (http://www.jmlr.org/papers/v11/mazumder10a.html)
 
 ## Installation
 As this package is not published in PyPI, you need to install directly from the Github
 repo. The following command installs the most recent version of the package:
 ```shell
pip install git+ssh://git@github.com/nimily/low-rank-impute.git@master#egg=impute
```  

## Usage
```python
import numpy as np
import numpy.random as npr
import numpy.linalg as npl

from impute import Dataset
from impute import SoftImpute
from impute import EntryTraceLinearOp

shape = 100, 100
n_row, n_col = shape

p = 0.05

b = np.ones(shape)

op = EntryTraceLinearOp(shape)
ys = []
for i in range(n_row):
    for j in range(n_col):
        if npr.binomial(1, p) == 1:
            op.append((i, j, 1))
            ys.append(b[i, j] + npr.randn())

ds = Dataset(op, ys)
imputer = SoftImpute(shape)

alpha_max = imputer.alpha_max(ds)
alpha_min = alpha_max / 100
alpha_seq = imputer.get_alpha_seq(ds, alpha_min, 0.5)

zs = imputer.fit(ds, alpha_seq)
ms = [z.to_matrix() for z in zs]

for alpha, m in zip(alpha_seq, ms):
    rel_err = npl.norm(b - m, 'fro') / npl.norm(b, 'fro')
    print('alpha = {alpha}, rel_err = {rel_err}'.format(alpha=alpha, rel_err=rel_err))
```

## Testing
You can test the package using the following shell command:
```shell
impute_path(){
    python -c 'import impute; print(impute.__path__[0])'
}
pytest "$impute_path"
```
