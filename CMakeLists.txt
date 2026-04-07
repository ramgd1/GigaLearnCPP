cmake_minimum_required (VERSION 3.8)

project("GigaLearnBot")

# Workaround: LibTorch's Caffe2 expects CUDA::nvToolsExt; newer CUDA/FindCUDAToolkit may not define it.
# Create the target if missing so find_package(Torch) can succeed (Linux + CUDA 12.x).
if(NOT TARGET CUDA::nvToolsExt)
	if(DEFINED CUDAToolkit_ROOT)
		set(_cuda_root "${CUDAToolkit_ROOT}")
	elseif(DEFINED ENV{CUDA_PATH})
		set(_cuda_root "$ENV{CUDA_PATH}")
	else()
		set(_cuda_root "/usr/local/cuda")
	endif()
	find_library(NVTOOLSEXT_LIB nvToolsExt NAMES nvToolsExt nvtoolsext
		PATHS "${_cuda_root}/lib64" "${_cuda_root}/lib" "/usr/lib/x86_64-linux-gnu" NO_DEFAULT_PATH)
	if(NVTOOLSEXT_LIB)
		add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
		set_target_properties(CUDA::nvToolsExt PROPERTIES IMPORTED_LOCATION "${NVTOOLSEXT_LIB}")
		message(STATUS "Created CUDA::nvToolsExt for LibTorch: ${NVTOOLSEXT_LIB}")
	else()
		# Dummy target so find_package(Torch) succeeds; link may still work if libtorch bundles it.
		add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
		message(STATUS "CUDA::nvToolsExt: library not found, using dummy target for LibTorch")
	endif()
	unset(_cuda_root)
endif()

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp")
add_executable(GigaLearnBot ${FILES_SRC})

# Set C++ version to 20
set_target_properties(GigaLearnBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnBot PROPERTIES CXX_STANDARD 20)

# Make sure GigaLearnCPP is going to build in the same directory as us
# Otherwise, we won't be able to import it at runtime
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Include RLGymSim_PPO
add_subdirectory(GigaLearnCPP)
target_link_libraries(GigaLearnBot GigaLearnCPP)

# Include RLBot
add_subdirectory(RLBotCPP)
target_link_libraries(GigaLearnBot RLBotCPP)