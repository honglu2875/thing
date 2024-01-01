# thing
Catch your tensors in one program and quietly send to another live python session.

![thing](https://upload.wikimedia.org/wikipedia/en/thumb/d/d0/Thing_%28The_Addams_Family%29.gif/375px-Thing_%28The_Addams_Family%29.gif)

# How it works
![thing_demo](assets/demo.gif)

# Quick start
Say, you have a neural network training script, and you just want to play with some tensors in a separate
python session, investigating the L2-norms, distributions, eigen-vectors, etc. in a live and interactive manner without
disturbing your training job.

## Client: catching the tensors
You can quickly modify your training code by inserting some `thing.catch(...)` as follows:
```python
import thing
import ... # your other imports
... # your codes
model = ... # you train some model
... # some more codes

for i in range(100):
    ... # your codes
    loss.backward()  # your backward pass
    optimizer.step()  # you applied your gradient
    
    # Now, it's show time:
    thing.catch(loss, every=10)
    thing.catch(model.lm_head.weight, name='lm_head_weight', every=10)

... # your rest of the codes
```

On another interactive python REPL, try
```
>>> import thing
>>> thing.serve()
```
Once you receive the captures, try
```
>>> print(thing.get('lm_head_weight'))
>>> print(thing.get('loss'))
```
to obtain the captured tensor in your server session. You can apply whatever transformation you want to investigate further.

## Server APIs: receive and interact

In the separate python REPL or jupyter notebook, other than `.get`, there are a few other APIs.

### help()

First of all, a full list of the APIs can be seen by doing `thing.help()`. I will not provide the full list here in this README.

### ingest_all()

Directly put all named variables inside your current scope. Meaning:

Say you did `thing.catch(loss)` on the training session. In the interactive session, you could do
```
>>> thing.serve()
2024-01-01 01:11:18,914 - thing.interactive - INFO     - Server started at port 2875.
>>> thing.ingest_all()
>>> loss
```
You will see the `loss` variable by directly calling like the above.

### status()

`thing.status()` will show you the current status with a spinner, but *you don't need to run this in order to receive the tensors*.

### summary()

`thing.summary()` prints the recent capture logs.


# FAQ
- Q: Why not logging?

  A: Logging is great if we know what metrics we are looking. But in the case of debugging and research, there is 
  data (especially big tensors such as layer weights) that we prefer to interactively explore in a separate persistent
  session.
  A few examples of my own use cases:

  - Debugging a model implementation. Quickly catching intermediate variables, keep them in a persistent python
    session, check the shapes and do some sanity tests.
  - Debugging distributed training (FSDP, etc). Just send stuff to a fixed ip.
  - Silently catch the hidden states in a continuous training/inference job for some quick analysis.
  
- Q: Why not using pickle?

  A: Tensors can be huge. I'd rather stream bytes directly from the original buffer address than doing serialization
  and make a copy of the whole thing in RAM.

- Q: What to do if the server receiving tensors is on a different machine?
  
  A: A few ways:

  - `ssh` with reverse port-forwarding.
  - Try to set the environment variable `THING_SERVER=<your-own-ip>`.
  - In the client, specify `thing.catch(..., server='<your-ip>:<your-port>')`

- Q: Does it work without specifying a name (such as calling `thing.catch(loss)`)?

  A: Yes. You will see on the other side by calling `thing.get("loss")`.

- Q: What's the point of specifying a name in: `.catch(..., name='...')`?

  A: A few things:
  - A name is **NOT** supposed to be a unique identifier of a transmission.
  - Several transmissions under the same name will have a chain of history on the server.
  - `thing.get_all(name)` returns a list of tensors, from oldest to latest.
  - By specifying a name, it avoids an expensive I/O to trace the variable name from the previous scope.