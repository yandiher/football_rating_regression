def model_comparison(x, y, scale=True):
    # import all important libraries
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    # condition if scale is true
    if scale == True:
        
        # scale the feature
        scale = StandardScaler()
        x_transform = scale.fit_transform(x)
        x = x_transform
        
        # split the data into train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
        
        # list all needed algorithms
        models = [RandomForestClassifier,
                  DecisionTreeClassifier,
                  KNeighborsClassifier,
                  SVC,
                  GaussianNB,
                  LogisticRegression,
                  SGDClassifier]
        
        # give name for the algorithms
        names = ['RandomForestClassifier', 
                 'DecisionTreeClassifier', 
                 'KNeighborsClassifier',
                 'SVC',
                 'GaussianNB',
                 'LogisticRegression',
                 'SGDClassifier']
        
        # store the score of the algorithms
        scores = []
    
        # iteration the models to get the best algorithm using the default parameter
        for m in models:
            model = m()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            score = accuracy_score(y_test, y_pred)
            scores.append(score)
        
        # store the result into dataframe and return the result
        result = pd.DataFrame({'Name':names, 'Score':scores})
        return result
    
    else:
        # split the data into train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
    
        # list all needed algorithms
        models = [RandomForestClassifier,
                  DecisionTreeClassifier,
                  KNeighborsClassifier,
                  SVC,
                  GaussianNB,
                  LogisticRegression,
                  SGDClassifier]
        
        # give name for the algorithms
        names = ['RandomForestClassifier', 
                 'DecisionTreeClassifier', 
                 'KNeighborsClassifier',
                 'SVC',
                 'GaussianNB',
                 'LogisticRegression',
                 'SGDClassifier']
    
        # store the score of the algorithms
        scores = []
    
        # iteration the models to get the best algorithm using the default parameter
        for m in models:
            model = m()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            score = accuracy_score(y_test, y_pred)
            scores.append(score)
        
        # store the result into dataframe and return the result
        result = pd.DataFrame({'Name':names, 'Score':scores})
        return result