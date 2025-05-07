LOAD CSV WITH HEADERS FROM "file:///articles.csv" as row  

WITH row 

CALL { 

    WITH row 

    MERGE(a:Article {id:row.article_id}) 

    SET a.desc = row.detail_desc 

    MERGE(p:Product {code:row.product_code}) 

    SET p.name = row.prod_name 

    MERGE(a)-[:OF_PRODUCT]->(p) 

    MERGE(pt:ProductType {id:row.product_type_no}) 

    SET pt.name = row.product_type_name 

    MERGE(p)-[:HAS_TYPE]->(pt) 

    WITH row, a, p 

    MERGE(pg:ProductGroup {name:row.product_group_name}) 

    MERGE(p)-[:HAS_GROUP]->(pg) 

    WITH row, a 

    MERGE(g:GraphicalAppearance {id:row.graphical_appearance_no}) 

    SET g.name = row.graphical_appearance_name 

    MERGE (a)-[:HAS_GRAPHICAL_APPEARANCE]->(g) 

    WITH row, a 

    MERGE (c:ColorGroup {id: row.colour_group_code}) 

    SET c.name = row.colour_group_name 

    MERGE (a)-[:HAS_COLOR]->(c) 

    WITH row, a 

    MERGE (pc:PerceivedColor {id: row.perceived_colour_value_id}) 

    SET pc.name = row.perceived_colour_value_name 

    MERGE (a)-[:HAS_PERCEIVED_COLOR]->(pc) 

    MERGE (pcm:PerceivedColor {id: row.perceived_colour_master_id}) 

    SET pcm.name = row.perceived_colour_master_name 

    MERGE (pc)-[:HAS_MASTER]->(pcm) 

    WITH row, a 

    MERGE (d:Department {id:row.department_no}) 

    SET d.name = row.department_name 

    MERGE (a)-[:HAS_DEPARTMENT]->(d) 

    WITH row, a 

    MERGE (i:Index {id: row.index_code}) 

    SET i.name = row.index_name 

    MERGE (a)-[:HAS_INDEX]->(i) 

    MERGE (ig:IndexGroup {id: row.index_group_no}) 

    SET ig.name = row.index_group_name 

    MERGE (i)-[:HAS_GROUP]->(ig) 

    WITH row, a 

    MERGE (s:Section {id: row.section_no}) 

    SET s.name = row.section_name 

    MERGE (a)-[:HAS_SECTION]->(s) 

    WITH row, a 

    MERGE (gg:GarmentGroup {id: row.garment_group_no}) 

    SET gg.name = row.garment_group_name 

    MERGE (a)-[:HAS_GARMENT_GROUP]->(gg) 

} IN TRANSACTIONS OF 1000 ROWS 

 