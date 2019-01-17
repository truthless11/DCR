# since the reference numbers have nothing to do with context, so we just treat them as symbols
# since the taxi type and taxi phone number is random generated, and have no relation with the user's requests. (in this dataset, users never require certain color or type of taxies)
# there is only one police station, so once the type of this slot is predicted, we get the true value.
# there is only one hospital, so once the type of this slot is predicted, we get the true value
symbol_list = {
    "[hotel_reference]",
    "[train_reference]",
    "[restaurant_reference]",
    "[attraction_reference]",
    "[hospital_reference]",
    "[taxi_type]",
    "[taxi_phone]",
    "[police_address]",
    "[police_phone]",
    "[police_postcode]",
    "[police_name]",
    "[hospital_postcode]",
    "[hospital_address]",
    "[hospital_name]",
    "[value_time]",
    "[value_price]",
    "[value_place]",
    "[value_day]",
    "[value_count]",
    "[train_id]"
}

# all the values in "entity_attr_list" almost only appear in system's utterence, so we remove them from user's utterence
# for which we only predict the value's type and corresponding entity, and then query from the database
entity_attr_list = {
    "[attraction_address]",
    "[restaurant_address]",
    "[attraction_phone]",
    "[restaurant_phone]",
    "[hotel_address]",
    "[restaurant_postcode]",
    "[attraction_postcode]",
    "[hotel_phone]",
    "[hotel_postcode]",
    "[hospital_phone]"
}

# the entities which should be queried from KG, it contains both relations and entities. Here we treat these relations (in fact the attributes of entities) also as entities.
entity_type_list = {
    # attr values (relations)
    "[value_area]",
    "[value_pricerange]",
    "[value_food]",
    # entity names
    "[hotel_name]",
    "[restaurant_name]",
    "[attraction_name]",
    "[hospital_department]"
}