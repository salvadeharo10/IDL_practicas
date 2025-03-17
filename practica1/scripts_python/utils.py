def trace_handler_cpu(p):
    output = p.key_averages().table(sort_by = "cpu_time_total", row_limit = 10)
    print(output)
    p.export_chrome_trace("/home/salvadordeharoo/IDL/practica1/trazas/" + str(p.step_num) + '.json')

def trace_handler_gpu(p):
    output = p.key_averages().table(sort_by = "self_cuda_time_total", row_limit = 10)
    print(output)
    p.export_chrome_trace("/home/salvadordeharoo/IDL/practica1/trazas/" + str(p.step_num) + '.json')