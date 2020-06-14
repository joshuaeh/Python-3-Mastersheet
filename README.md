# Python-3-Mastersheet
Notes on Python3, Git, and a number of useful packages   
Flexible input is often denoted with underscores```varbatim code``` ```_to be filled in_```  
# Python
## Slicing
```a[#]``` Select specific element  
```a[x:y:z]``` Select elements starting at x up to but not including y at step z. (may be negative to count backward)
    -z may be implicit.
    -```a[x:]``` slice starting at x  
    -```a[:y]``` slice up to but not including y  
```:``` Select all within dimension  
```..``` For as many dots, represents ```:,:```  
## Read/write
```open(_file_, _mode_)``` because this is unsafe to opena and not close, use with preceeding with clause
    -```'r'``` read
    -```'a'``` append
    -```'w'``` write
    -```'x'``` create
# Git bash commands
```cd _relative path_``` change cwd  
```pwd``` print current working directory  
```ld``` list contents of directory  
### Setup
```git config --global user.name "[*firstname lastname*]"``` set a name for yourself   
```git init``` initialize cwd as a git repository    
```git clone _url_``` retrieve an entire repository from url  
### Stage and Snapshot
```git status``` show modified files in cwd  
```git add```  
    - ```.``` add all files not ignored to the commit and begin tracking  
    - ```-a``` stage all tracked changes to commit  
```git reset _file_``` unstage a file from commit  
```git diff``` what is changed but not staged  
```git diff --staged``` what is staged but not committed  
```git commit -m "_descriptive message_"``` commit staged content with message  
### Update
```git remote add _alias_ _url_``` add git url as an alias  
```git fetch _alias_``` fetches down all branches from remote   
```git merge _alias_/_branch_``` marbe a remote branch into current branch and bring it up to date  
```git push _alias_ _branch_``` Transmit local branch  commits to the remote repo  
```git pull``` fetch and merge any commits from the tracked repo  
```git log``` show the commit history of the active branch  

# Packages
The real strength of Python is the access to lots of good packages.

# Installing and managing an environment
I prefer to use pip to manage and update packages.  
```pip freeze``` show what is currently installed in current environment  
```pip install _package_``` install package into current environment  
    -```pip install _package_==_version number_``` set installation to version of package  
    -```pip install git+_url_@_branch name_``` if a repo is configured to be a package, installs repo as package  
```pip uninstall _package_```  

# Numpy
C backend and lots of array and vectorization functions allow this to be ideal for numerical calculation.  
## Array Creation and manipulation
Basic object within numpy is a multidimensional array. The 0 axis, or dimension, will be the highest level and each subsequent axis will be the nect dimension in. Length is the number or elements within each dimension. Ex. a 2d Array. The 0 axis has a length 3, the 1 axis has length 5. (or 2 and 5 respectively if we start at 0 pythonically)
```python
a = array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
```  
```a = np.array([1, 2, 3, 4], dtype=_type_)``` Initialize an array with optional type declaration 
```np.zeros((_shape_))``` create an array of zeros to specified shape  
```np.ones((_shape_))``` create an array of ones in shape specified  
```np.arange(#)``` create a 1-d array with # elements counting from 0
### Modification methods
```a.T``` Transpose list or array  
```a.shape``` return shape  
```a.ndim``` return number of dimensions  
```a.reshape(_new shape_)``` reshape array. note that the number of elements must match  
```a.flat``` flatten multidimensional array.  
### Array operations
A and B are 2 2x2 arrays.  
```A*B``` elementwise multiplication  
```A@B``` Matrix multiplication  
```A.dot(B)``` dot product
## load and Save
Save/load pickled file: ```np.save()``` ```np.load()```  
Save/load text file: ```np.loadtxt()``` ```np.savetxt()```  
## polynomials
```np.polyfit(x, y, _degree_)``` 
## Statistics
mean ```np.mean()```  
standard deviation ```np.std()```
median ```np.median()```  
```np.random```  
    ```.rand(n)``` n random numbers between 0 and 1  
    ```.normal(mean, std, count)```  
    ```.seed()``` Helps create repeatable random numbers

# Scipy
## .integrate

## .stats
```skew``` reports skew  

# Pandas
```df=pd.DataFrame()``` Initialize dataframe
## load/save
```df.to_csv('_name_.csv', index=False)``` Save to csv  
```df.read_csv('_name_.csv')```  
```df.to_json('_name_.json',orient='table',index=False)```  
```df.to_excel('_name_.xlsx',index=False)``` 
## Analysis
```.describe()``` simple statistics  
```.head()``` display first 5 lines  


# Graphing
## Matplotlib
normal Stuff:
```python
plt.subplot(_rows_, _columns_, _current figure_)
ax.grid()
plt.legend()
plt.show()
```
Misc Plots:
```python
plt.scatter(x, y)
plt.hist(_var_, _sins_, alpha=_transparency_, label='_name_')
```
**c=Colors**

    =============    ===============================
    character        color
    =============    ===============================
    ``'b'``          blue
    ``'g'``          green
    ``'r'``          red
    ``'y'``          yellow
    ``'k'``          black
    =============    ===============================

**m=Markers**

    =============    ===============================
    character        description
    =============    ===============================
    ``'.'``          point marker
    ``'o'``          circle marker
    ``'s'``          square marker
    ``'^'``          triangle marker
    ``'*'``          star marker
    =============    ===============================

**ln=Line Styles**

    =============    ===============================
    character        description
    =============    ===============================
    ``'-'``          solid line style
    ``'--'``         dashed line style
    ``'-.'``         dash-dot line style
    ``':'``          dotted line style
    =============    ===============================
## plotly
Interactive html-based interactive plotting. 
### plotly express
```python
import plotly.express as px
```
#### Scatter Plot
```python
fig = px.scatter(data,x='x',y='y',color='w',size='x',hover_data=['w'])
fig.show()
```
### plotly graph_objects

## seaborn
wrapper for matplotlib that makes things prettier
```python
import seaborn as sns
```
Misc Plots:
```python
sns.pairplot(data[['x','y','z','w']],hue=('w'))  # pairplot to observe trends with variables
sns.boxplot(x='w',y='x',data=data)               # box plot
sns.violinplot(x='w',y='x',data=data,size=6)     # violin plot
sns.jointplot('x','z',data=data,kind="kde")      # shows two variables and univariate joint distributions. cool heat map between                                                            distributinos of both

```

# Analysis
## sklearn
### .preprocessing
common scaling methods
```s = StandardScaler()``` Each column is normalized to zero mean and standard deviation of one  
```fit_transform(x)``` Fit and transform  
```transform(x)``` transform based on another fit  
```inverse_transform(xs)``` Scale back to original representation
### .metrics
```r2_score(meas, model)```
### .model_selection
```train_test_split``` split data for training and testing
    ```train,test = train_test_split(ds, test_size=0.2, shuffle=True)```
## Statsmodels
performs analysis with nice summary
```import statsmodels.api as sm```
```python
xc = sm.add_constant(x)          # initialize constant
model = sm.OLS(z,xc).fit()
predictions = model.predict(xc)
model.summary()
```
