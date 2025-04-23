MATCH (c:Customer)-[:SUMMER_2019]->(start), (c)-[:FALL_2019]->()<-[:NEXT]-(end) 

WITH c, start, end SKIP 100 LIMIT 1 

MATCH p=(start)-[:NEXT*]->(end) 

WITH nodes(p) as nodes, relationships(p) as rels 

UNWIND nodes as node  

MATCH p=(node)-[:HAS_ARTICLE]->(a) 

RETURN a.desc as article 