
function test()
	forwardhiddenstack = Any[]
	push!(forwardhiddenstack, "beginining")
	push!(forwardhiddenstack, "mid")
	push!(forwardhiddenstack, "end")
	reverse!(forwardhiddenstack)
	backwardhiddenstack = Any[];
	push!(backwardhiddenstack, "beginining")
	push!(backwardhiddenstack, "mid")
	push!(backwardhiddenstack, "end")

	@assert length(forwardhiddenstack) == length(backwardhiddenstack)
	global ypreds=Any[]
	i=1
	while !isempty(forwardhiddenstack)
		fh = pop!(forwardhiddenstack)
		bh = pop!(backwardhiddenstack)
		println("ypred[$i] --> $fh + $bh")
		push!(ypreds, i)
		i += 1
	end
end

function test2()
	ygolds = ["ygold1", "ygold2", "ygold3"]
	gfs = Any[]
	gbs = Any[]
	for i=length(ygolds):-1:1
		push!(gfs, ygolds[i])
		push!(gbs, ygolds[i])
	end

	println("First pop --> $(pop!(gfs))")
	println("Second pop --> $(pop!(gfs))")
	println("Third pop --> $(pop!(gfs))")

	println("======= if I reverse ===========")
	for i=length(ygolds):-1:1
		push!(gfs, ygolds[i])
		push!(gbs, ygolds[i])
	end
	reverse!(gfs)

	println("First pop --> $(pop!(gfs))")
	println("Second pop --> $(pop!(gfs))")
	println("Third pop --> $(pop!(gfs))")
end
