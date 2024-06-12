add_library(src
        # core
        core/core.cuh
        core/c_abcore.cpp
        core/c_abcore_decomp.cpp

        # graph folder
        graph/graph.h
        graph/graph.cpp

        # util folder
        util/config.h
        util/dbg.h
        util/log.cpp
        util/log.h
        util/table.h
        util/timer.cuh
        util/utility.h
        util/uf.h
)

# Link libraries (if needed)
target_link_libraries(src PUBLIC argparse)

# Include directories
target_include_directories(src PUBLIC
        ${argparse_SOURCE_DIR}
        )
# target_include_directories(src PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
