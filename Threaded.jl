module Threaded

export @threaded

macro threaded(expr)
   esc(quote
       if Threads.nthreads() == 1
           $(expr)
       else
           Threads.@threads $(expr)
       end
   end)
end

end
