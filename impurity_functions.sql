cur.execute("CREATE FUNCTION bayes_error(p numeric) RETURNS numeric AS $$ SELECT LEAST(p, 1 - p); $$ LANGUAGE SQL;")
cur.execute("CREATE FUNCTION gini(p numeric) RETURNS numeric AS $$ SELECT p*(1-p); $$ LANGUAGE SQL;")
cur.execute("CREATE FUNCTION cross_entropy(p numeric) RETURNS numeric AS $$ SELECT -1*p*LOG(p) - (1-p)*LOG(1-p); $$ LANGUAGE SQL;")
