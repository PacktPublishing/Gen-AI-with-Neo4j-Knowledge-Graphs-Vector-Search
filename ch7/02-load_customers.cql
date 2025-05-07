:auto 
LOAD CSV WITH HEADERS FROM "file:///customers.csv" as row  

WITH row 

CALL { 

    WITH row  

    MERGE (c:Customer {id:row.customer_id}) 

    SET c.age = row.age  

    FOREACH( ignoreME in CASE WHEN row.fashion_news_frequency = 'Regularly'  THEN [1] ELSE [] END | 

        SET c:FN_REGULAR 

    ) 

    FOREACH( ignoreME in CASE WHEN row.club_member_status = 'ACTIVE'  THEN [1] ELSE [] END | 

        SET c:CLUB_ACTIVE 

    ) 

    FOREACH( ignoreME in CASE WHEN row.club_member_status = 'PRE-CREATE'  THEN [1] ELSE [] END | 

        SET c:CLUB_PRE_CREATE 

    ) 

    FOREACH( ignoreME in CASE WHEN row.Active <> 'ACTIVE'  THEN [1] ELSE [] END | 

        SET c:INACTIVE 

    ) 

    MERGE(p:PostalCode {code:row.postal_code}) 

    MERGE(c)-[:LIVES_IN]->(p) 

} IN TRANSACTIONS OF 1000 ROWS 

 