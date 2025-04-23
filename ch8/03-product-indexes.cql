CREATE CONSTRAINT product_code_idx FOR (n:Product) REQUIRE n.code IS UNIQUE ; 

CREATE CONSTRAINT article_id_idx FOR (n:Article) REQUIRE n.id IS UNIQUE ; 

CREATE CONSTRAINT product_type_id_idx FOR (n:ProductType) REQUIRE n.id IS UNIQUE ; 

CREATE CONSTRAINT colour_group_idx FOR (n:ColorGroup) REQUIRE n.id IS UNIQUE ; 

CREATE CONSTRAINT product_group_name_idx FOR (n:ProductGroup) REQUIRE n.name IS UNIQUE ; 

CREATE CONSTRAINT graphical_appearance_id_idx FOR (n:GraphicalAppearance) REQUIRE n.id IS UNIQUE ; 

CREATE CONSTRAINT perceived_colour_id_idx FOR (n:PerceivedColor) REQUIRE n.id IS UNIQUE ; 

CREATE CONSTRAINT department_id_idx FOR (n:Department) REQUIRE n.id IS UNIQUE ; 

CREATE CONSTRAINT section_id_idx FOR (n:Section) REQUIRE n.id IS UNIQUE ; 

CREATE CONSTRAINT garment_group_id_idx FOR (n:GarmentGroup) REQUIRE n.id IS UNIQUE ; 

CREATE CONSTRAINT article_index_id_idx FOR (n:Index) REQUIRE n.id IS UNIQUE ; 

CREATE CONSTRAINT article_index_group_id_idx FOR (n:IndexGroup) REQUIRE n.id IS UNIQUE ; 
