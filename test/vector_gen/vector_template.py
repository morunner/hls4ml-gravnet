template_str = """
#pragma once
#include <string>
#include <vector>
#include <cstddef>
{% for class_name, items in data.items() %}
struct {{ class_name }} {
    std::string name;
{%- for field in get_fields(items[0]) %}
    {%- if is_array(items[0], field) %}
    const float* {{ field }};
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
static const float {{ class_name }}_data_{{ loop.index0 }}_{{ field }}[] = {{ cpp(get_attr(item, field)) }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}
{%- endfor %}
{% for class_name, items in data.items() %}
static const {{ class_name }} vectors_{{ class_name }}[] = {
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
