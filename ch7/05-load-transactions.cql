LOAD CSV WITH HEADERS FROM "file:///transactions_train.csv" as row WITH row 
CALL { 

    WITH row 

    MATCH (c:Customer {id:row.customer_id}) 

    MATCH (a:Article {id:row.article_id}) 

    WITH a, c, row 

    CREATE (t:Transaction {date: row.t_dat, price: row.price, salesChannel: row.sales_channel_id}) 

    CREATE (t)-[:HAS_ARTICLE]->(a) 

    WITH c, t 

    CALL { 

        WITH c, t 

        WITH c, t 

        WHERE exists((c)-[:START_TRANSACTION]->()) OR exists((c)-[:LATEST]->())  

        MATCH (c)-[r:LATEST]->(lt) 

        DELETE r  

        CREATE (lt)-[:NEXT]->(t) 

        CREATE (c)-[:LATEST]->(t) 

        UNION 

        WITH c, t  

        WITH c,t 

        WHERE NOT ( exists((c)-[:START_TRANSACTION]->()) OR exists((c)-[:LATEST]->()) ) 

        CREATE (c)-[:START_TRANSACTION]->(t) 

        CREATE (c)-[:LATEST]->(t) 

    } 

} IN TRANSACTIONS OF 1000 ROWS 

 