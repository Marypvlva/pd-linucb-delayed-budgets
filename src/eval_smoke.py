import numpy as np

z = np.load("data/processed/obd_feedback.npz")
X, A, R, C, D = z["X"], z["A"], z["R"], z["C"], z["D"]

print("X", X.shape, X.dtype)
print("A", A.shape, A.dtype, "unique actions:", len(np.unique(A)), "min/max:", int(A.min()), int(A.max()))
print("R", R.shape, R.dtype, "mean:", float(R.mean()), "sum:", float(R.sum()))
print("C", C.shape, C.dtype, "sum:", float(C.sum()))
print("D", D.shape, D.dtype, "unique:", np.unique(D)[:10])
print("keys:", z.files)
