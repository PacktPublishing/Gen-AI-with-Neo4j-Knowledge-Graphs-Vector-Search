CREATE CONSTRAINT customer_id_idx FOR (n:Customer) REQUIRE n.id IS UNIQUE ; 

CREATE CONSTRAINT postal_code_idx FOR (n:PostalCode) REQUIRE n.code IS UNIQUE ; 

