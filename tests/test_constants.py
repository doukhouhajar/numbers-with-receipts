from groundwork.tools import lookup_constant, compute_quantity, dimension_of

print(lookup_constant("speed_of_light"))       # Constant with a CODATA source
print(lookup_constant("made_up_thing"))        # Unknown
print(dimension_of("us"))                      # [time]

c = lookup_constant("speed_of_light")
q = compute_quantity("light_ms", "result = speed_of_light * 1e-3",
                     {"speed_of_light": c}, unit="m")
print(q)                                        # Derived, with code_ref