import os

def get_trace_handler_cpu(model_name):
    def trace_handler_cpu(p):
        step = p.step_num
        os.makedirs("/home/salvadordeharoo/IDL/practica1/trazas", exist_ok=True)

        output = p.key_averages().table(sort_by="cpu_time_total", row_limit=20)
        print(output)
        

    return trace_handler_cpu


def get_trace_handler_gpu(model_name):
    def trace_handler_gpu(p):
        step = p.step_num
        os.makedirs("/home/salvadordeharoo/IDL/practica1/trazas", exist_ok=True)

        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
        print(output)
        
        
        p.export_chrome_trace(f"/home/salvadordeharoo/IDL/practica1/trazas/trace_gpu_{model_name}_step{step}.json")
    return trace_handler_gpu