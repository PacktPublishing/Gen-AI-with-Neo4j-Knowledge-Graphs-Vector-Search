MATCH (c:Customer) 

WITH c 

CALL { 

    WITH c 

    MATCH (c)-[:START_TRANSACTION]->(s) 

    MATCH (c)-[:LATEST]->(e) 

    WITH c,s,e 

    MATCH p=(s)-[:NEXT*]->(e) 

    WITH c, nodes(p) as nodes 

    UNWIND nodes as node 

    WITH c, node, node.date as d  

    WITH c, node, toInteger(substring(d, 0,4)) as year, substring(d, 5,2) as month 

    WITH c, node,  

        CASE WHEN month="12" THEN year  

             WHEN month="01" OR month="02" THEN year-1 

            ELSE  

                year  

        END as year, 
        CASE WHEN month="12" OR month="01" OR month="02" THEN "WINTER" 

             WHEN month="03" OR month="04" OR month="05" THEN "SPRING" 

             WHEN month="06" OR month="07" OR month="08" THEN "SUMMER" 

             WHEN month="09" OR month="10" OR month="11" THEN "FALL" 

        END as season 

    WITH c, node, season+'_'+year as relName 
    WITH c, relName, head(collect(node)) as start 

    WHERE relName is not null 

    CALL apoc.create.relationship(c, relName, {}, start) YIELD rel 

    WITH 1 as out  

    return DISTINCT out 

} IN TRANSACTIONS OF 1000 ROWS 

WITH 1 as r  

RETURN DISTINCT r 

 