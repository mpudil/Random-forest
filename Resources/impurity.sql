-- Impurity functions, implemented as SQL functions.
-- Alex Reinhart

-- First, we make an aggregate function to calculate p, the fraction of entries
-- in a column that are 1.

CREATE AGGREGATE fraction_ones (numeric)
(
        sfunc = fraction_ones_accum,
        stype = numeric[],
        finalfunc = fraction_ones_div,
        initcond = '{0,0}'
);


-- Next, we implement the entropies using this function.

CREATE FUNCTION bayes_error(p numeric) RETURNS numeric AS $$
       SELECT LEAST(p, 1 - p);
$$ LANGUAGE SQL;
