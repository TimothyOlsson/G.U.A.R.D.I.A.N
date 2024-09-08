from glob import glob

for js_file in glob("guardian_website\pkg\*.js"):

    # Load file
    with open(js_file, 'r') as fp:
        content = fp.read()

    # Issue with older wgpu (<= 0.12.0)
    content = content.replace("getObject(arg0).dispatch(", "getObject(arg0).dispatchWorkgroups(")

    # Issue with newer wgpu (>= 0.13.0)
    content = content.replace("getObject(arg0).end();", "getObject(arg0).endPass();")

    # Issue with newer wgpu (>= 0.13.0)
    content = content.replace("const ret = getObject(arg0).createComputePipeline(getObject(arg1));",
                              '\n'.join(["const shader = getObject(arg1);",
                                         "delete shader.layout;",
                                         "const ret = getObject(arg0).createComputePipeline(shader);"]))

    with open(js_file, 'w') as fp:
        fp.write(content)