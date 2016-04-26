import pstats, cProfile
import sys
import pyximport; pyximport.install()
import complex_features_c_optimized as copt
import numpy as np
from IPython import get_ipython
ipython = get_ipython()



# list1 = np.array([1,2,3,1,1,1,4,3,3,2,2,2,2,2,2,2,2])
list1 = np.array([1,1,1,-1,1,1,-4,3,3,1,1,1,1,1,1,1,1], dtype=np.float64)
print list1
print copt.expanding_diff(list1,0)
diff = copt.expanding_abs_min_diff_diff(list1)
print diff
# totals, percentages, mapping = copt.expanding_total_and_percentage(list1)
# print totals, percentages, mapping


#need to make sure expanding mode gets lists with at least 2 elements (or change this in future)
# lists = []
# for i in xrange(1000):
    # a = np.random.randint(2,10, size=1)
    # b = np.random.randint(200, size=a[0])
    # lists.append(','.join(map(str,b)))


# small_lists = []
# for i in xrange(10):
    # a = np.random.randint(2,5, size=1)
    # b = np.random.randint(20, size=a[0])
    # small_lists.append(','.join(map(str,b)))
# small_lists.append(small_lists[-1])
# print small_lists
# stability = copt.expanding_jitter_ohe(small_lists)
# print jitter


# small_lists = ['2', '2', '2', '2', '2', '2', '3']
# print small_lists
# totals, percentages, mapping = copt.expanding_total_and_percentage_ohe(small_lists)
# print totals, percentages, mapping
#num_lists = np.random.randint(200, size=100000)
#ipython.magic("timeit copt.expanding_mode(np.array([3,3,1,5,2,2,4,4,4,4,4,4]))")
#ipython.magic("timeit copt.expanding_mode_py(np.array([3,3,1,5,2,2,4,4,4,4,4,4]))")
#cProfile.runctx("ipython.magic('timeit copt.expanding_mode(a)')",globals(),locals(),"Profile.prof")
#cProfile.runctx("copt.expanding_mode_ohe(lists)",globals(),locals(),"Profile.prof")
#cProfile.runctx("copt.expanding_total_and_percentage(num_lists)",globals(),locals(),"Profile.prof")
#cProfile.runctx("copt.expanding_total_and_percentage_ohe(lists)",globals(),locals(),"Profile.prof")

#s = pstats.Stats("Profile.prof")
#s.strip_dirs().sort_stats("time").print_stats()
