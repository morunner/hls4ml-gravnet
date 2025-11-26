template_str = """
#pragma once
#include <string>
#include <cstddef>

struct gravnet_config {
    static const unsigned B = {{ gravnet_config.B }};
    static const unsigned V = {{ gravnet_config.V }};
    static const unsigned F = {{ gravnet_config.F }};
    static const unsigned S = {{ gravnet_config.S }};
    static const unsigned n_neighbors = {{ gravnet_config.n_neighbors }};
    static const unsigned exp_table_size = {{ gravnet_config.exp_table_size }};
    static const unsigned exp_table_size_nbits = {{ gravnet_config.exp_table_size_nbits }};
    static const unsigned exp_table_indexing_shmt = {{ gravnet_config.exp_table_indexing_shmt }};
};
{% for class_name, items in data.items() %}
struct {{ class_name }} {
    std::string name;
{%- for field in get_fields(items[0]) %}
    {%- if is_array(items[0], field) %}
    float* {{ field }};
    size_t {{ field }}_len;
    {%- else %}
    {{ get_cpp_type(items[0], field) }} {{ field }};
    {%- endif %}
{%- endfor %}
};
{%- endfor %}
{% for class_name, items in data.items() %}
    {%- for item in items %}
        {%- for field in get_fields(item) %}
            {%- if is_array(item, field) %}
static float {{ class_name }}_data_{{ loop.index0 }}_{{ field }}[] = {{ cpp(get_attr(item, field)) }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}
{%- endfor %}
{% for class_name, items in data.items() %}
static const int {{ class_name }}s_length = {{ nvectors(class_name) }};
static {{ class_name }} {{ class_name }}s[] = {
    {%- for item in items %}
    {
        "{{ item.name }}",
        {%- for field in get_fields(item) %}
            {%- if is_array(item, field) %}
        {{ class_name }}_data_{{ loop.index0 }}_{{ field }},
        {{ length(get_attr(item, field)) }},
            {%- else %}
        {{ get_attr(item, field) }},
            {%- endif %}
        {%- endfor %}
    },
    {%- endfor %}
};
{%- endfor %}
"""
