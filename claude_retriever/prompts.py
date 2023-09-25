def retrieval_prompt(tool_name, tool_description, user_input):
    prompt = f"""

    Human: You're going to talk with a human. Please be helpful, honest, and harmless. Please make sure to think carefully, step-by-step when answering the human. 

    During this conversation you have access to a set of tools. You can invoke a tool by writing a "<function_calls>" block like the following as part of your reply to the user:
    <function_calls>
    <invoke>
    <tool_name>$TOOL_NAME</tool_name>
    <parameters>
    <$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
    ...
    </parameters>
    </invoke>
    </function_calls>

    The output and/or any errors will appear in a subsequent "<function_results>" block, and remain there as part of your reply to the user.
    You may then continue composing the rest of your reply to the user, respond to any errors, or make further function calls as appropriate.
    If a "<function_results>" does NOT appear after your function calls, then they are likely malformatted and not recognized as a call.

    Here are the tools available:
    ```
    <tools>

    <tool_description>
    <tool_name>{tool_name}</tool_name>
    <description>
    {tool_description}
    </description>
    <parameters>
    <parameter>
    <name>query</name>
    <type>str</type>
    <description>The query to provide to the {tool_name}.</description>
    </parameter>
    </parameters>
    </tool_description>

    </tools>
    ```
    <strategy>
    Please perform a series of searches to find the most relevant information to answer this query. Some tips for good searches:
    - Do not use questions in the searches, only keywords
    - Start with a broad search, then narrow down to more specific searches
    After each </result></function_results> do the following:
    - Within <search_quality_reflection> tags, reflect on whether the original query could be satisfactorily answered 
    by the cumulative search results so far, and what kinds of extra information you might still need to answer the query.
    - Then within <search_quality_score> tags, provide a rating between 1 and 5, where 5 means the query could be answered very well using all the search results so far, and 1 means the query could not be answered at all.
    -- If search_quality_score < 4, please try again with a different search query.
    -- If search_quality_score >= 4, please give the human a final answer to the question, and finish. 
    -- Please put your final answer inside of <result> tags. If after several search results you still determine that 
    you don't have enough information to answer the question, you can just write 'I don't know' inside of the <result> tags.

    Please make sure not to make up facts that aren't supported by the search results.
    </strategy>

    {user_input}

    Assistant:
    """

    return prompt

def citations_prompt(search_results, user_input):
    prompt = f"""
    
    Human: You're going to talk with a human. Please be helpful, honest, and harmless. Please make sure to think carefully, step-by-step when answering the human.
    <strategy>
    If you are provided with search results in the following format:

    <search_results>
    <item index="1">
    <source>
    (a unique identifying source for this item - could be a URL, file name, hash, etc)
    </source>
    <content>
    (the content of the document - could be a passage, web page, article, etc)
    </content>
    </item>
    <item index="2">
    <source>
    (a unique identifying source for this item - could be a URL, file name, hash, etc)
    </source>
    <content>
    (the content of the document - could be a passage, web page, article, etc)
    </content>
    </item>
    ...
    </item>

    And a query related to the search results, then you should provide an answer to the query based on the information in 
    the documents, and use the following citation format precisely:

    - Citations and Bibliography: If your answer is based on anything in the search results you MUST put <cite></cite> 
    tags inline, around the content that is supported by any of the search results. Inside each cite tag, you must have
    a <display_text> tag and a <bib_reference> tag. If you use any citation tags, you must also output a bibliography 
    at the end of your answer, which should be wrapped in <bibliography> tags. The bibliography must must contain a 
    series of <bib_item> tags, one per citation. Each <bib_item> tag must contain a <bib_reference> tag, a <source> 
    tag, and a <quote> tag.

    Here are some more details on the sub-tags that go inside the <cite> tags:
        - The <bib_reference> should be a reference to an item in your bibliography. This can just be an integer.
        - The <display_text> attribute will be stripped out and shown to the user inline, alongside the rest of your 
    answer, and your response should flow nicely when the whole <cite> tag is just replaced with the <display_text>. 

    Here are some more details on the sub-tags that go inside the <bib_item> tags:
        - The <bib_reference> should be a tag that labels each bibliography item with a specific reference number. This
    can just be an integer. Make sure that the <bib_reference> matches one of the <bib_reference> tags in a <cite> 
    citation tag. This is how bib_items and citations are matched with each other.
        - The <quote> tag should contain an exact, word-for-word quote from one of the search results.
        - The <source> should be the exact source that the quote is pulled from. Search results will contain <source> 
    tags, and you should copy the <source> tags from those tags.

    Note that you should output exactly one <bib_item> tag per <cite> tag. I.e. there must be a one-to-one relationship
    between <bib_item> tags and <cite> tags.

    - Other than giving citations, do not mention or make reference to the search results in any fashion in your answer.
    - Make sure to put your citations in your final answer, which should be in <result> tags.

    </strategy>    
    
    {search_results}

    Query:
    {user_input}
    
    Assistant:
    """

    return prompt